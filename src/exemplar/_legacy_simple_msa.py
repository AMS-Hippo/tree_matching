import math
import random
from collections import Counter

NEG_INF = float("-inf")


# ----------------------------
# Utilities
# ----------------------------
def log_add(a, b):
    """Stable log(exp(a)+exp(b)) for scalars."""
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if b > a:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def estimate_pi(L, alpha=0.5):
    """
    Empirical distribution over tokens in L with add-alpha smoothing.
    Returns dict token -> prob.
    """
    counts = Counter(tok for seq in L for tok in seq)
    alphabet = list(counts.keys())
    if not alphabet:
        return {}
    total = sum(counts.values()) + alpha * len(alphabet)
    return {tok: (counts[tok] + alpha) / total for tok in alphabet}


def weighted_choice(rng, items, weights_dict):
    """Sample one item from items using probabilities in weights_dict."""
    r = rng.random()
    cum = 0.0
    for x in items:
        cum += weights_dict[x]
        if r <= cum:
            return x
    return items[-1]


def move_set(length, min_len, max_len):
    moves = []
    if length < max_len:
        moves.append("insert")
    if length > min_len:
        moves.append("delete")
    if length >= 2:
        moves.append("swap")
    if length >= 1:
        moves.append("replace")
    return moves


def adjust_to_bounds(seq, min_len, max_len, rng, alphabet, pi):
    """Truncate/pad a sequence to satisfy length bounds."""
    seq = list(seq)
    if len(seq) > max_len:
        seq = seq[:max_len]
    while len(seq) < min_len:
        seq.append(weighted_choice(rng, alphabet, pi))
    return seq


# ----------------------------
# Likelihood: P(obs | master)
# ----------------------------
def log_prob_obs_given_master(obs, master, p_skip, p_add, pi, eps=1e-12):
    """
    Forward DP for a simple skip/insert model:

    At any point you may:
      - insert a random token ~ pi with prob p_add (does not advance master)
      - otherwise (prob 1-p_add), if master remains:
           skip current master token with prob p_skip
           emit current master token with prob 1-p_skip (must match obs)

    After finishing master, stop with prob (1-p_add). (A constant factor in M.)
    """
    if not (0 <= p_add < 1):
        raise ValueError("Require 0 <= p_add < 1")
    if not (0 <= p_skip <= 1):
        raise ValueError("Require 0 <= p_skip <= 1")

    m = len(master)
    n = len(obs)

    log_p_add = math.log(p_add) if p_add > 0 else NEG_INF
    log_p_skip = math.log(p_skip) if p_skip > 0 else NEG_INF
    log_not_add = math.log1p(-p_add)     # log(1 - p_add)
    log_emit = math.log1p(-p_skip)       # log(1 - p_skip)

    dp = [[NEG_INF] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0

    for i in range(m + 1):
        for j in range(n + 1):
            cur = dp[i][j]
            if cur == NEG_INF:
                continue

            # Insert obs[j] (doesn't advance i)
            if j < n and p_add > 0:
                tok = obs[j]
                pj = pi.get(tok, eps)
                dp[i][j + 1] = log_add(dp[i][j + 1], cur + log_p_add + math.log(pj))

            # If master remains: either skip it, or emit it (must match obs[j])
            if i < m:
                if p_skip > 0:
                    dp[i + 1][j] = log_add(dp[i + 1][j], cur + log_not_add + log_p_skip)

                if j < n and obs[j] == master[i]:
                    dp[i + 1][j + 1] = log_add(dp[i + 1][j + 1], cur + log_not_add + log_emit)

    # stop after finishing master (one final "not-insert" decision)
    return dp[m][n] + log_not_add


def total_log_likelihood(L, M, p_skip, p_add, pi, idxs=None):
    if idxs is None:
        seqs = L
    else:
        seqs = (L[i] for i in idxs)

    s = 0.0
    for obs in seqs:
        s += log_prob_obs_given_master(obs, M, p_skip, p_add, pi)
    return s


# ----------------------------
# MCMC moves (with Hastings correction)
# ----------------------------
def propose_move(M, rng, alphabet, pi, min_len, max_len, log_pi=None):
    """
    Returns (M_new, log_q_forward, log_q_reverse) for MH acceptance.
    Tokens are proposed from pi.
    """
    if log_pi is None:
        log_pi = {k: math.log(v) for k, v in pi.items()}

    m = len(M)
    moves = move_set(m, min_len, max_len)
    if not moves:
        raise ValueError("No legal moves: check min_len/max_len.")

    move = rng.choice(moves)

    if move == "insert":
        pos = rng.randrange(m + 1)
        tok = weighted_choice(rng, alphabet, pi)
        M_new = M[:pos] + [tok] + M[pos:]

        log_q_f = -math.log(len(moves)) - math.log(m + 1) + log_pi[tok]

        moves2 = move_set(len(M_new), min_len, max_len)
        # reverse is delete at same pos
        log_q_r = -math.log(len(moves2)) - math.log(len(M_new))
        return M_new, log_q_f, log_q_r

    if move == "delete":
        pos = rng.randrange(m)
        tok = M[pos]
        M_new = M[:pos] + M[pos + 1:]

        log_q_f = -math.log(len(moves)) - math.log(m)

        moves2 = move_set(len(M_new), min_len, max_len)
        # reverse is insert tok back at pos
        log_q_r = -math.log(len(moves2)) - math.log(len(M_new) + 1) + log_pi.get(tok, math.log(1e-12))
        return M_new, log_q_f, log_q_r

    if move == "swap":
        a = rng.randrange(m)
        b = rng.randrange(m - 1)
        if b >= a:
            b += 1
        if a > b:
            a, b = b, a

        M_new = M.copy()
        M_new[a], M_new[b] = M_new[b], M_new[a]

        num_pairs = m * (m - 1) / 2
        log_q = -math.log(len(moves)) - math.log(num_pairs)
        return M_new, log_q, log_q

    if move == "replace":
        pos = rng.randrange(m)
        tok_new = weighted_choice(rng, alphabet, pi)
        tok_old = M[pos]

        M_new = M.copy()
        M_new[pos] = tok_new

        log_q_f = -math.log(len(moves)) - math.log(m) + log_pi[tok_new]
        log_q_r = -math.log(len(moves)) - math.log(m) + log_pi.get(tok_old, math.log(1e-12))
        return M_new, log_q_f, log_q_r

    raise RuntimeError("Unknown move type")


def run_mcmc(
    L, M0, p_skip, p_add, pi,
    min_len, max_len, n_steps, rng,
    eval_batch=None,
    temperature=1.0
):
    """
    Runs MH on a fixed subset of L for speed (eval_batch), tracks best state,
    and returns (best_M, full_loglik(best_M)).
    """
    n = len(L)
    idxs = None if (eval_batch is None or eval_batch >= n) else rng.sample(range(n), eval_batch)

    alphabet = list(pi.keys())
    if not alphabet:
        raise ValueError("Alphabet empty; is L empty?")

    log_pi = {k: math.log(v) for k, v in pi.items()}

    M = adjust_to_bounds(M0, min_len, max_len, rng, alphabet, pi)
    cur_ll = total_log_likelihood(L, M, p_skip, p_add, pi, idxs=idxs)

    best_M = M
    best_ll = cur_ll

    for _ in range(n_steps):
        M_prop, log_q_f, log_q_r = propose_move(M, rng, alphabet, pi, min_len, max_len, log_pi=log_pi)
        prop_ll = total_log_likelihood(L, M_prop, p_skip, p_add, pi, idxs=idxs)

        log_alpha = (prop_ll - cur_ll) / temperature + (log_q_r - log_q_f)
        if math.log(rng.random()) < min(0.0, log_alpha):
            M, cur_ll = M_prop, prop_ll
            if cur_ll > best_ll:
                best_M, best_ll = M, cur_ll

    full_ll = total_log_likelihood(L, best_M, p_skip, p_add, pi, idxs=None)
    return best_M, full_ll


def infer_master_sequence(
    L,
    p_skip=0.05,
    p_add=0.02,
    G=None,
    min_len=1,
    max_len=50,
    n_steps=5000,
    n_restarts=5,
    eval_batch=300,     # None => full likelihood each step (can be slow)
    temperature=1.0,
    seed=None,
):
    """
    Main entry point. Input: list-of-lists of strings. Output: list of strings.
    """
    if not L:
        return []

    pi = estimate_pi(L, alpha=0.5)
    alphabet = list(pi.keys())
    rng = random.Random(seed)

    if G is None:
        G = list(rng.choice(L))
    G = adjust_to_bounds(G, min_len, max_len, rng, alphabet, pi)

    global_best = None
    global_best_ll = NEG_INF

    for r in range(n_restarts):
        M0 = G.copy()
        if r > 0:
            # diversify start by applying a few random moves (no accept/reject)
            for _ in range(5 + 5 * r):
                M0, _, _ = propose_move(M0, rng, alphabet, pi, min_len, max_len)

        M_r, ll_r = run_mcmc(
            L, M0, p_skip, p_add, pi,
            min_len, max_len,
            n_steps=n_steps,
            rng=rng,
            eval_batch=eval_batch,
            temperature=temperature,
        )
        if ll_r > global_best_ll:
            global_best_ll = ll_r
            global_best = M_r

    return global_best
