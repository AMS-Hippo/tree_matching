
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class ModelConfig:
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int
    max_seq_len: int
    d_model: int = 128
    num_seq_layers: int = 1
    num_dec_layers: int = 1
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    min_support_size: int = 2
    max_support_size: int = 6
    batch_size: int = 32
    epochs: int = 10
    episodes_per_epoch: int = 1000
    lr: float = 3e-4
    weight_decay: float = 1e-4
    lambda_len: float = 0.5
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    verbose: bool = True
    num_torch_threads: int = 1


class EpisodicSequenceClassDataset(Dataset):
    """
    Dataset of meta-learning style episodes.

    Each item is:
      - a support set from one label/class
      - a held-out target sequence from the same label/class
    """

    def __init__(
        self,
        grouped_sequences: Mapping[Hashable, Sequence[Sequence[int]]],
        min_support_size: int = 2,
        max_support_size: int = 6,
        episodes_per_epoch: int = 1000,
        seed: int = 0,
    ) -> None:
        self.grouped_sequences: Dict[Hashable, List[List[int]]] = {
            label: [list(seq) for seq in seqs]
            for label, seqs in grouped_sequences.items()
            if len(seqs) >= 2
        }
        if len(self.grouped_sequences) == 0:
            raise ValueError("Need at least one label/class with at least 2 sequences.")

        self.labels = list(self.grouped_sequences.keys())
        self.min_support_size = min_support_size
        self.max_support_size = max_support_size
        self.episodes_per_epoch = episodes_per_epoch
        self.seed = seed

    def __len__(self) -> int:
        return self.episodes_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng = random.Random(self.seed + idx)
        label = self.labels[rng.randrange(len(self.labels))]
        seqs = self.grouped_sequences[label]

        target_idx = rng.randrange(len(seqs))
        available_idxs = [i for i in range(len(seqs)) if i != target_idx]
        max_k = min(self.max_support_size, len(available_idxs))
        if max_k < 1:
            raise ValueError(f"Label {label!r} has too few examples for episodic training.")

        min_k = min(self.min_support_size, max_k)
        k = rng.randint(min_k, max_k)
        support_idxs = rng.sample(available_idxs, k=k)

        support_sequences = [list(seqs[i]) for i in support_idxs]
        target_sequence = list(seqs[target_idx])

        return {
            "label": label,
            "support_sequences": support_sequences,
            "target_sequence": target_sequence,
        }


def make_collate_fn(pad_id: int, bos_id: int, eos_id: int):
    def collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        batch_size = len(batch)
        max_k = max(len(item["support_sequences"]) for item in batch)
        max_support_len = max(
            max(len(seq) + 1 for seq in item["support_sequences"]) for item in batch
        )  # +EOS

        support = torch.full(
            (batch_size, max_k, max_support_len), pad_id, dtype=torch.long
        )
        support_slot_mask = torch.zeros((batch_size, max_k), dtype=torch.bool)

        for b, item in enumerate(batch):
            for j, seq in enumerate(item["support_sequences"]):
                support_slot_mask[b, j] = True
                seq_with_eos = list(seq) + [eos_id]
                support[b, j, : len(seq_with_eos)] = torch.tensor(seq_with_eos, dtype=torch.long)

        max_tgt_len = max(len(item["target_sequence"]) + 1 for item in batch)  # +EOS
        tgt_in = torch.full((batch_size, max_tgt_len), pad_id, dtype=torch.long)
        tgt_out = torch.full((batch_size, max_tgt_len), pad_id, dtype=torch.long)
        tgt_len = torch.zeros(batch_size, dtype=torch.long)

        labels: List[Any] = []
        raw_targets: List[List[int]] = []
        for b, item in enumerate(batch):
            target = list(item["target_sequence"])
            raw_targets.append(target)
            labels.append(item["label"])

            in_ids = [bos_id] + target
            out_ids = target + [eos_id]

            tgt_in[b, : len(in_ids)] = torch.tensor(in_ids, dtype=torch.long)
            tgt_out[b, : len(out_ids)] = torch.tensor(out_ids, dtype=torch.long)
            tgt_len[b] = len(out_ids)

        return {
            "support": support,
            "support_slot_mask": support_slot_mask,
            "tgt_in": tgt_in,
            "tgt_out": tgt_out,
            "tgt_len": tgt_len,
            "labels": labels,
            "raw_targets": raw_targets,
        }

    return collate


class SetConditionedGenerator(nn.Module):
    """
    Lightweight, CPU-friendly set-conditioned generator.

    Architecture:
      1. Encode each support sequence with a GRU.
      2. Aggregate the support-set with learned attention pooling.
      3. Predict an explicit output-length distribution from the class context.
      4. Decode autoregressively with a GRU, conditioned on:
            - the class context
            - the requested output length

    This is intentionally simpler than a full transformer so that the
    example notebook stays runnable on ordinary CPUs.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.max_seq_len < 2:
            raise ValueError("max_seq_len must be at least 2.")

        self.config = config
        self.pad_id = config.pad_id
        self.bos_id = config.bos_id
        self.eos_id = config.eos_id
        self.max_seq_len = config.max_seq_len
        self.d_model = config.d_model

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.embedding_dropout = nn.Dropout(config.dropout)

        self.seq_encoder = nn.GRU(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=config.num_seq_layers,
            batch_first=True,
            dropout=config.dropout if config.num_seq_layers > 1 else 0.0,
        )

        self.set_score = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.Tanh(),
            nn.Linear(config.d_model, 1),
        )

        self.len_emb = nn.Embedding(config.max_seq_len + 1, config.d_model)
        self.len_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.max_seq_len + 1),
        )

        self.decoder_gru = nn.GRU(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=config.num_dec_layers,
            batch_first=True,
            dropout=config.dropout if config.num_dec_layers > 1 else 0.0,
        )
        self.init_h = nn.Linear(2 * config.d_model, config.num_dec_layers * config.d_model)
        self.out_head = nn.Linear(config.d_model, config.vocab_size)

    def _check_supported_length(self, T: int) -> None:
        if T > self.max_seq_len:
            raise ValueError(
                f"Encountered sequence length {T}, but model max_seq_len is {self.max_seq_len}."
            )

    def encode_support(
        self,
        support: torch.Tensor,
        support_slot_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        support:
            [B, K, Ts] integer token ids, each real support sequence should end in EOS
        support_slot_mask:
            [B, K] boolean, True where a support slot is real

        Returns:
            context: [B, D]
            seq_repr: [B, K, D]
            support_slot_mask: [B, K]
        """
        B, K, Ts = support.shape
        self._check_supported_length(Ts)

        flat = support.reshape(B * K, Ts)
        flat_slot_mask = support_slot_mask.reshape(B * K)

        tok_valid = flat.ne(self.pad_id)
        lengths = tok_valid.sum(dim=1)

        # Avoid zero-length sequences in padded support slots.
        if (~flat_slot_mask).any():
            flat = flat.clone()
            lengths = lengths.clone()
            missing = ~flat_slot_mask
            flat[missing, 0] = self.eos_id
            lengths[missing] = 1

        emb = self.embedding_dropout(self.token_emb(flat))
        packed = pack_padded_sequence(
            emb,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, h = self.seq_encoder(packed)
        seq_repr = h[-1].view(B, K, self.d_model)

        scores = self.set_score(seq_repr).squeeze(-1)
        scores = scores.masked_fill(~support_slot_mask, -1e9)
        attn = torch.softmax(scores, dim=1)
        context = (attn.unsqueeze(-1) * seq_repr).sum(dim=1)

        return context, seq_repr, support_slot_mask

    def forward(
        self,
        support: torch.Tensor,
        support_slot_mask: torch.Tensor,
        tgt_in: torch.Tensor,
        tgt_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        support:            [B, K, Ts]
        support_slot_mask:  [B, K]
        tgt_in:             [B, Tt]
        tgt_len:            [B]  full output length including EOS
        """
        Tt = tgt_in.size(1)
        self._check_supported_length(Tt)

        context, _, _ = self.encode_support(
            support=support,
            support_slot_mask=support_slot_mask,
        )
        len_logits = self.len_head(context)

        len_cond = self.len_emb(tgt_len)  # [B, D]
        dec_in = self.embedding_dropout(self.token_emb(tgt_in))
        dec_in = dec_in + context.unsqueeze(1) + len_cond.unsqueeze(1)

        init_h = torch.tanh(self.init_h(torch.cat([context, len_cond], dim=-1)))
        init_h = init_h.view(
            self.config.num_dec_layers,
            tgt_in.size(0),
            self.d_model,
        )

        dec_out, _ = self.decoder_gru(dec_in, init_h)
        logits = self.out_head(dec_out)
        return logits, len_logits


def generator_loss(
    logits: torch.Tensor,
    tgt_out: torch.Tensor,
    len_logits: torch.Tensor,
    tgt_len: torch.Tensor,
    pad_id: int,
    lambda_len: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tgt_out.reshape(-1),
        ignore_index=pad_id,
    )
    length_loss = F.cross_entropy(len_logits, tgt_len)
    total = token_loss + lambda_len * length_loss
    return total, {
        "token_loss": float(token_loss.detach().cpu()),
        "length_loss": float(length_loss.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }


def _move_batch_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def train_model(
    grouped_sequences: Mapping[Hashable, Sequence[Sequence[int]]],
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> Tuple[SetConditionedGenerator, List[Dict[str, float]]]:
    """
    Train the set-conditioned generator from grouped integer-token sequences.
    """
    set_seed(training_config.seed)
    if training_config.num_torch_threads is not None and training_config.num_torch_threads > 0:
        torch.set_num_threads(training_config.num_torch_threads)

    model = SetConditionedGenerator(model_config).to(training_config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.lr,
        weight_decay=training_config.weight_decay,
    )

    dataset = EpisodicSequenceClassDataset(
        grouped_sequences=grouped_sequences,
        min_support_size=training_config.min_support_size,
        max_support_size=training_config.max_support_size,
        episodes_per_epoch=training_config.episodes_per_epoch,
        seed=training_config.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        collate_fn=make_collate_fn(
            pad_id=model_config.pad_id,
            bos_id=model_config.bos_id,
            eos_id=model_config.eos_id,
        ),
    )

    history: List[Dict[str, float]] = []
    for epoch in range(1, training_config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_token_loss = 0.0
        total_len_loss = 0.0
        total_batches = 0

        for batch in loader:
            batch = _move_batch_to_device(batch, training_config.device)
            optimizer.zero_grad(set_to_none=True)

            logits, len_logits = model(
                support=batch["support"],
                support_slot_mask=batch["support_slot_mask"],
                tgt_in=batch["tgt_in"],
                tgt_len=batch["tgt_len"],
            )
            loss, stats = generator_loss(
                logits=logits,
                tgt_out=batch["tgt_out"],
                len_logits=len_logits,
                tgt_len=batch["tgt_len"],
                pad_id=model_config.pad_id,
                lambda_len=training_config.lambda_len,
            )
            loss.backward()
            if training_config.grad_clip is not None and training_config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip)
            optimizer.step()

            total_loss += stats["total_loss"]
            total_token_loss += stats["token_loss"]
            total_len_loss += stats["length_loss"]
            total_batches += 1

        epoch_stats = {
            "epoch": float(epoch),
            "avg_total_loss": total_loss / max(total_batches, 1),
            "avg_token_loss": total_token_loss / max(total_batches, 1),
            "avg_length_loss": total_len_loss / max(total_batches, 1),
        }
        history.append(epoch_stats)

        if training_config.verbose:
            print(
                f"Epoch {epoch:>3d} | "
                f"total={epoch_stats['avg_total_loss']:.4f} | "
                f"token={epoch_stats['avg_token_loss']:.4f} | "
                f"length={epoch_stats['avg_length_loss']:.4f}"
            )

    return model, history


def prepare_support_batch(
    support_sequences: Sequence[Sequence[int]],
    pad_id: int,
    eos_id: int,
    device: str | torch.device = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(support_sequences) == 0:
        raise ValueError("support_sequences must contain at least one sequence.")

    K = len(support_sequences)
    Ts = max(len(seq) + 1 for seq in support_sequences)  # +EOS
    support = torch.full((1, K, Ts), pad_id, dtype=torch.long, device=device)
    support_slot_mask = torch.ones((1, K), dtype=torch.bool, device=device)

    for j, seq in enumerate(support_sequences):
        seq_with_eos = list(seq) + [eos_id]
        support[0, j, : len(seq_with_eos)] = torch.tensor(
            seq_with_eos, dtype=torch.long, device=device
        )

    return support, support_slot_mask


@torch.no_grad()
def predict_length_distribution(
    model: SetConditionedGenerator,
    support_sequences: Sequence[Sequence[int]],
) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    support, support_slot_mask = prepare_support_batch(
        support_sequences=support_sequences,
        pad_id=model.pad_id,
        eos_id=model.eos_id,
        device=device,
    )
    context, _, _ = model.encode_support(support, support_slot_mask)
    probs = model.len_head(context).softmax(dim=-1)[0]
    return probs.detach().cpu()


@torch.no_grad()
def greedy_decode_exact_len(
    model: SetConditionedGenerator,
    support_sequences: Sequence[Sequence[int]],
    out_len: int,
) -> List[int]:
    """
    Greedy decode exactly `out_len` tokens, where `out_len` includes EOS.
    """
    if out_len < 1:
        raise ValueError("out_len must be at least 1.")

    model.eval()
    device = next(model.parameters()).device
    support, support_slot_mask = prepare_support_batch(
        support_sequences=support_sequences,
        pad_id=model.pad_id,
        eos_id=model.eos_id,
        device=device,
    )

    prefix = torch.tensor([[model.bos_id]], dtype=torch.long, device=device)
    for _ in range(out_len):
        logits, _ = model(
            support=support,
            support_slot_mask=support_slot_mask,
            tgt_in=prefix,
            tgt_len=torch.tensor([out_len], dtype=torch.long, device=device),
        )
        next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
        prefix = torch.cat([prefix, next_tok], dim=1)

    return prefix[0, 1:].detach().cpu().tolist()


@torch.no_grad()
def generate_representatives(
    model: SetConditionedGenerator,
    support_sequences: Sequence[Sequence[int]],
    n_outputs: int = 3,
) -> List[Dict[str, Any]]:
    """
    Return a few representative sequences for a class.

    Strategy:
      1. Predict the class-conditioned length distribution.
      2. Take the top lengths.
      3. Decode one greedy sequence for each such length.
    """
    length_probs = predict_length_distribution(model, support_sequences)
    usable = length_probs[1:]  # ignore length 0
    topk = min(n_outputs, usable.numel())
    top_lengths = torch.topk(usable, k=topk).indices + 1

    outputs: List[Dict[str, Any]] = []
    for L in top_lengths.tolist():
        token_ids = greedy_decode_exact_len(model, support_sequences, out_len=L)
        outputs.append(
            {
                "predicted_length": int(L),
                "token_ids": token_ids,
                "length_probability": float(length_probs[L].item()),
            }
        )
    return outputs


def save_checkpoint(
    path: str,
    model: SetConditionedGenerator,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    token_to_id: Dict[str, int],
    id_to_token: Dict[int, str],
    history: Optional[Sequence[Dict[str, float]]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "token_to_id": dict(token_to_id),
        "id_to_token": dict(id_to_token),
        "history": list(history) if history is not None else None,
        "extra_metadata": dict(extra_metadata) if extra_metadata is not None else None,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    model_config = ModelConfig(**payload["model_config"])
    training_config = TrainingConfig(**payload["training_config"])
    model = SetConditionedGenerator(model_config)
    model.load_state_dict(payload["state_dict"])
    model.to(map_location)
    model.eval()

    payload["model_config"] = model_config
    payload["training_config"] = training_config
    payload["model"] = model
    return payload
