from __future__ import annotations

"""Likelihood-style exemplar inference wrappers.

This module wraps two legacy exemplar solvers from the old tree-alignment code:

- a lightweight local-search / MCMC ascent method, and
- a stronger indel-model workflow with EM-like updates and optional MCMC polish.

The wrappers adapt these solvers to the new :class:`SequenceBag` interface,
provide medoid / POA-based initialization, and expose benchmark-friendly
metadata.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .bags import SequenceBag
from .medoid import infer_medoid_sequence
from .poa import infer_poa_sequence
from .preprocess import prepare_sequences_from_bag
from . import _legacy_simple_msa as _simple
from . import _legacy_chatgpt_msa as _workflow


@dataclass
class LikelihoodResult:
    exemplar: List[Any]
    backend: str
    score: float
    init_method: str = 'na'
    metadata: Dict[str, Any] = field(default_factory=dict)


def _stringify_sequences(seqs: Sequence[Sequence[Any]]) -> List[List[str]]:
    return [[str(tok) for tok in seq] for seq in seqs]


def _temporary_bag_from_sequences(seqs: Sequence[Sequence[Any]]) -> SequenceBag:
    return SequenceBag.from_sequences([tuple(seq) for seq in seqs], deduplicate=True, name='prepared_sequences')


def _choose_init_sequence(
    bag: SequenceBag,
    prepared_seqs: Sequence[Sequence[Any]],
    *,
    init_mode: str = 'medoid',
    poa_backend: str = 'progressive',
    poa_seed_mode: str = 'medoid',
    poa_order_mode: str = 'closest_first',
) -> List[Any]:
    mode = str(init_mode).lower().strip()
    if mode == 'medoid':
        return list(infer_medoid_sequence(_temporary_bag_from_sequences(prepared_seqs)))
    if mode == 'poa':
        tmp_bag = _temporary_bag_from_sequences(prepared_seqs)
        return list(
            infer_poa_sequence(
                tmp_bag,
                backend=poa_backend,
                seed_mode=poa_seed_mode,
                order_mode=poa_order_mode,
                repeat_mode='unique',
                target_total_repeats=tmp_bag.n_unique,
                max_total_repeats=max(tmp_bag.n_unique, 1),
                denoise_rounds=0,
                return_result=False,
            )
        )
    if mode == 'longest':
        if not prepared_seqs:
            return []
        return list(max(prepared_seqs, key=len))
    if mode == 'shortest':
        if not prepared_seqs:
            return []
        return list(min(prepared_seqs, key=len))
    if mode == 'first':
        return list(prepared_seqs[0]) if prepared_seqs else []
    raise ValueError("init_mode must be 'medoid', 'poa', 'longest', 'shortest', or 'first'")


def _infer_length_window(
    prepared_seqs: Sequence[Sequence[Any]],
    init_seq: Sequence[Any],
    *,
    lower_slack: int = 3,
    upper_slack: int = 8,
    min_len_floor: int = 1,
    max_len_cap: Optional[int] = None,
) -> Dict[str, Any]:
    if prepared_seqs:
        lens = np.asarray([len(seq) for seq in prepared_seqs], dtype=int)
        obs_min = int(lens.min())
        obs_med = int(np.median(lens))
        obs_max = int(lens.max())
    else:
        obs_min = obs_med = obs_max = 0
    init_len = int(len(init_seq))
    target_len = max(obs_max, init_len, obs_med)
    min_len = max(int(min_len_floor), min(obs_min, init_len or obs_min or 1) - int(lower_slack))
    max_len = target_len + int(upper_slack)
    if max_len_cap is not None:
        max_len = min(max_len, int(max_len_cap))
    max_len = max(max_len, max(target_len, min_len))
    lengths = list(range(int(min_len), int(max_len) + 1))
    return {
        'obs_min_len': int(obs_min),
        'obs_median_len': int(obs_med),
        'obs_max_len': int(obs_max),
        'init_len': int(init_len),
        'min_len': int(min_len),
        'target_len': int(target_len),
        'max_len': int(max_len),
        'lengths': lengths,
    }


def _resize_encoded_init(
    init_enc: np.ndarray,
    target_len: int,
    *,
    rng: np.random.Generator,
    bg_probs: np.ndarray,
) -> np.ndarray:
    cur = np.asarray(init_enc, dtype=np.int32).copy()
    L = int(target_len)
    if cur.size == L:
        return cur
    if cur.size > L:
        return np.asarray(cur[:L], dtype=np.int32)
    bg = np.asarray(bg_probs, dtype=np.float64)
    bg = bg / bg.sum()
    pad = rng.choice(bg.shape[0], size=L - cur.size, p=bg).astype(np.int32)
    return np.concatenate([cur, pad]).astype(np.int32)


def _run_seeded_workflow(
    prepared_seqs: Sequence[Sequence[Any]],
    init_seq: Sequence[Any],
    *,
    seed: int = 0,
    p_ins: Optional[float] = None,
    p_del: Optional[float] = None,
    eps_sub: Optional[float] = None,
    fix_provided_params: bool = True,
    bg_pseudocount: float = 1e-2,
    estimate_bg_from_inserts: bool = True,
    estimate_eps_sub: bool = True,
    eps_max: float = 0.30,
    n_restarts: int = 4,
    em_iters: int = 8,
    do_mcmc_refine: bool = True,
    mcmc_steps_refine: int = 2500,
    mcmc_temperature: float = 0.7,
    search_subset_size: int = 2000,
    progress: bool = False,
    lower_slack: int = 3,
    upper_slack: int = 8,
    min_len_floor: int = 1,
    max_len_cap: Optional[int] = None,
) -> LikelihoodResult:
    if not prepared_seqs:
        return LikelihoodResult(exemplar=[], backend='workflow_seeded', score=0.0, init_method='empty')

    lens_info = _infer_length_window(
        prepared_seqs,
        init_seq,
        lower_slack=lower_slack,
        upper_slack=upper_slack,
        min_len_floor=min_len_floor,
        max_len_cap=max_len_cap,
    )
    rng = np.random.default_rng(int(seed))
    alphabet = _workflow.Alphabet.from_sequences(prepared_seqs)
    obs_all = [alphabet.encode(s) for s in prepared_seqs]
    N = int(len(obs_all))
    if N > int(search_subset_size):
        idx = rng.choice(N, size=int(search_subset_size), replace=False)
        obs_search = [obs_all[int(i)] for i in idx]
    else:
        obs_search = obs_all

    bg_model = _workflow.BackgroundModel.fit(prepared_seqs, alphabet=alphabet, pseudocount=float(bg_pseudocount))
    bg_init = np.asarray(bg_model.probs, dtype=np.float64)

    if p_ins is None or not fix_provided_params:
        p_ins_fixed = None
    else:
        p_ins_fixed = float(p_ins)
    if p_del is None or not fix_provided_params:
        p_del_fixed = None
    else:
        p_del_fixed = float(p_del)
    if eps_sub is None or not fix_provided_params:
        eps_fixed = None
    else:
        eps_fixed = float(eps_sub)

    init_enc = alphabet.encode(init_seq) if init_seq else np.zeros(0, dtype=np.int32)
    best_score = float('-inf')
    best_master = None
    best_params = None
    best_hist = None

    for L in lens_info['lengths']:
        p_del0 = float(p_del) if p_del is not None else 0.20
        p_ins0 = float(p_ins) if p_ins is not None else _workflow._estimate_p_ins_from_lengths(
            float(np.mean([len(s) for s in prepared_seqs])), int(L), p_del0
        )
        eps0 = float(eps_sub) if eps_sub is not None else 0.02
        params0 = _workflow.IndelParams(
            p_ins=float(np.clip(p_ins0, 1e-4, 0.9)),
            p_del=float(np.clip(p_del0, 1e-4, 0.9)),
            bg_probs=bg_init,
            eps_sub=float(eps0),
        )
        init_for_L = _resize_encoded_init(init_enc, int(L), rng=rng, bg_probs=bg_init)

        for r in range(int(max(1, n_restarts))):
            if r == 0:
                master0 = init_for_L.copy()
            elif r == 1:
                master0 = _workflow._choose_init_master_from_data(obs_search, target_len=int(L), rng=rng)
                if int(master0.size) != int(L):
                    master0 = _resize_encoded_init(master0, int(L), rng=rng, bg_probs=bg_init)
            else:
                master0 = rng.integers(0, alphabet.size, size=int(L), dtype=np.int32)

            m_em, p_em, hist = _workflow._em_fixed_length(
                obs=obs_search,
                master=master0,
                params=params0,
                n_iters=int(em_iters),
                estimate_bg_from_inserts=bool(estimate_bg_from_inserts),
                estimate_eps_sub=bool(estimate_eps_sub),
                bg_pseudocount=float(bg_pseudocount),
                eps_max=float(eps_max),
                progress=bool(progress),
                fixed_p_ins=p_ins_fixed,
                fixed_p_del=p_del_fixed,
                fixed_eps_sub=eps_fixed,
                fixed_bg_probs=bg_init if (fix_provided_params and False) else None,
            )
            lik = _workflow.IndelLikelihood(p_em)
            ll = 0.0
            for o in obs_search:
                v = lik.forward_loglik(m_em, o)
                if not np.isfinite(v):
                    ll = float('-inf')
                    break
                ll += float(v)
            if float(ll) > float(best_score):
                best_score = float(ll)
                best_master = np.asarray(m_em, dtype=np.int32).copy()
                best_params = p_em
                best_hist = hist

    assert best_master is not None and best_params is not None

    if bool(do_mcmc_refine):
        est = _workflow.MasterSequenceEstimator(
            alphabet=alphabet,
            sequences=prepared_seqs,
            params=best_params,
            seed=int(seed),
        )
        mcmc_res = est.run_mcmc(
            init_master=best_master,
            n_steps=int(mcmc_steps_refine),
            burn_in=max(100, int(mcmc_steps_refine // 5)),
            thin=max(5, int(mcmc_steps_refine // 100)),
            fixed_length=True,
            temperature=float(mcmc_temperature),
            progress=bool(progress),
        )
        best_master = np.asarray(mcmc_res.best_master, dtype=np.int32).copy()
    else:
        mcmc_res = None

    master_final, params_final, hist_full = _workflow._em_fixed_length(
        obs=obs_all,
        master=best_master,
        params=best_params,
        n_iters=max(1, int(em_iters)),
        estimate_bg_from_inserts=bool(estimate_bg_from_inserts),
        estimate_eps_sub=bool(estimate_eps_sub),
        bg_pseudocount=float(bg_pseudocount),
        eps_max=float(eps_max),
        progress=False,
        fixed_p_ins=p_ins_fixed,
        fixed_p_del=p_del_fixed,
        fixed_eps_sub=eps_fixed,
        fixed_bg_probs=None,
    )
    lik_full = _workflow.IndelLikelihood(params_final)
    full_ll = 0.0
    for o in obs_all:
        v = lik_full.forward_loglik(master_final, o)
        if not np.isfinite(v):
            full_ll = float('-inf')
            break
        full_ll += float(v)

    return LikelihoodResult(
        exemplar=alphabet.decode(master_final),
        backend='workflow_seeded',
        score=float(full_ll),
        init_method='seeded',
        metadata={
            **lens_info,
            'n_input_sequences': int(len(prepared_seqs)),
            'search_subset_size_used': int(len(obs_search)),
            'best_search_subset_loglik': float(best_score),
            'do_mcmc_refine': bool(do_mcmc_refine),
            'mcmc_accept_rate': float(mcmc_res.accept_rate) if mcmc_res is not None else None,
            'em_history_search': best_hist,
            'em_history_full': hist_full,
            'p_ins_final': float(params_final.p_ins),
            'p_del_final': float(params_final.p_del),
            'eps_sub_final': float(params_final.eps_sub),
        },
    )


def infer_likelihood_sequence(
    bag: SequenceBag,
    *,
    engine: str = 'workflow_seeded',
    init_mode: str = 'medoid',
    init_sequence: Optional[Sequence[Any]] = None,
    sequence_repeat_mode: str = 'weights',
    target_total_sequences: int = 96,
    max_total_sequences: int = 192,
    denoise_rounds: int = 0,
    denoise_seed: int = 0,
    lower_slack: int = 3,
    upper_slack: int = 8,
    min_len_floor: int = 1,
    max_len_cap: Optional[int] = None,
    poa_backend: str = 'progressive',
    poa_seed_mode: str = 'medoid',
    poa_order_mode: str = 'closest_first',
    seed: int = 0,
    return_result: bool = False,
    **kwargs: Any,
) -> LikelihoodResult | List[Any]:
    if bag.n_unique == 0:
        out = LikelihoodResult(exemplar=[], backend=str(engine), score=0.0, init_method='empty', metadata={'n_input_sequences': 0})
        return out if return_result else out.exemplar

    prepared = prepare_sequences_from_bag(
        bag,
        repeat_mode=sequence_repeat_mode,
        target_total_sequences=target_total_sequences,
        max_total_sequences=max_total_sequences,
        denoise_rounds=denoise_rounds,
        denoise_seed=denoise_seed,
    )
    if init_sequence is not None:
        init_seq = list(init_sequence)
        resolved_init_method = 'provided'
    else:
        init_seq = _choose_init_sequence(
            bag,
            prepared,
            init_mode=init_mode,
            poa_backend=poa_backend,
            poa_seed_mode=poa_seed_mode,
            poa_order_mode=poa_order_mode,
        )
        resolved_init_method = str(init_mode)

    eng = str(engine).lower().strip()
    if eng in {'simple', 'simple_ascent', 'local_mcmc'}:
        lens_info = _infer_length_window(
            prepared,
            init_seq,
            lower_slack=lower_slack,
            upper_slack=upper_slack,
            min_len_floor=min_len_floor,
            max_len_cap=max_len_cap,
        )
        seqs = _stringify_sequences(prepared)
        init = [str(tok) for tok in init_seq]
        p_skip = float(kwargs.pop('p_skip', 0.08))
        p_add = float(kwargs.pop('p_add', 0.03))
        n_steps = int(kwargs.pop('n_steps', 12000))
        n_restarts = int(kwargs.pop('n_restarts', 4))
        eval_batch = kwargs.pop('eval_batch', 512)
        temperature = float(kwargs.pop('temperature', 0.9))
        best = _simple.infer_master_sequence(
            seqs,
            p_skip=p_skip,
            p_add=p_add,
            G=init,
            min_len=int(lens_info['min_len']),
            max_len=int(lens_info['max_len']),
            n_steps=n_steps,
            n_restarts=n_restarts,
            eval_batch=eval_batch,
            temperature=temperature,
            seed=int(seed),
        )
        pi = _simple.estimate_pi(seqs, alpha=0.5)
        score = _simple.total_log_likelihood(seqs, best, p_skip, p_add, pi, idxs=None)
        res = LikelihoodResult(
            exemplar=list(best),
            backend='simple',
            score=float(score),
            init_method=resolved_init_method,
            metadata={
                **lens_info,
                'n_input_sequences': int(len(prepared)),
                'sequence_repeat_mode': sequence_repeat_mode,
                'denoise_rounds': int(denoise_rounds),
                'p_skip': float(p_skip),
                'p_add': float(p_add),
                'n_steps': int(n_steps),
                'n_restarts': int(n_restarts),
                'eval_batch': None if eval_batch is None else int(eval_batch),
                'temperature': float(temperature),
                'init_len': int(len(init_seq)),
            },
        )
        return res if return_result else res.exemplar

    if eng in {'workflow', 'workflow_seeded', 'em'}:
        res = _run_seeded_workflow(
            prepared,
            init_seq,
            seed=int(seed),
            lower_slack=lower_slack,
            upper_slack=upper_slack,
            min_len_floor=min_len_floor,
            max_len_cap=max_len_cap,
            **kwargs,
        )
        res.init_method = resolved_init_method
        res.metadata.update({
            'sequence_repeat_mode': sequence_repeat_mode,
            'denoise_rounds': int(denoise_rounds),
            'init_len': int(len(init_seq)),
        })
        return res if return_result else res.exemplar

    raise ValueError("engine must be 'simple' or 'workflow_seeded'")


def infer_cluster_likelihood_exemplars(
    bags: Mapping[int, SequenceBag],
    *,
    return_details: bool = False,
    **kwargs: Any,
) -> Dict[int, LikelihoodResult] | Dict[int, List[Any]]:
    results = {int(c): infer_likelihood_sequence(bag, return_result=True, **kwargs) for c, bag in bags.items()}
    if return_details:
        return results
    return {int(c): list(res.exemplar) for c, res in results.items()}


__all__ = [
    'LikelihoodResult',
    'infer_likelihood_sequence',
    'infer_cluster_likelihood_exemplars',
]
