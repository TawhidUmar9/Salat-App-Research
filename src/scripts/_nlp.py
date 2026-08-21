"""
Transformer inference helpers.
==============================
Shared by 02b (sentiment) and 03/03b (ABSA + promise extraction).

Two things live here:

  * `SequenceClassifier` — batched sequence classification with length
    bucketing, autocast, and OOM backoff.

  * `ZeroShotScorer` — an NLI zero-shot classifier tuned for the scale this
    project needs. A naive `pipeline("zero-shot-classification")` run costs
    one forward pass per (sentence x label) pair; over millions of sentences
    and 27 aspect labels that is intractable. This implementation cuts the
    cost roughly two orders of magnitude via:

      1. exact-duplicate collapsing (short review sentences repeat heavily),
      2. an embedding prefilter that shortlists the top-k plausible labels
         per sentence before the expensive NLI model sees it,
      3. length-bucketed batching so padding waste stays near zero,
      4. entailment-logit-only scoring in bf16/fp16 autocast.

    The prefilter is a recall/cost tradeoff and must be reported as such:
    `prefilter_recall()` measures what it drops against a labelled set.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np

from _common import autocast_dtype, log, run_with_oom_backoff, sort_by_length


# ─── Sequence classification ────────────────────────────────────────────────────

@dataclass
class SequenceClassifier:
    """Batched HF sequence classifier returning full probability matrices."""

    model_name: str
    device: str = "cpu"
    batch_size: int = 32
    max_length: int = 256
    fp32: bool = False
    _tok: object = field(default=None, repr=False)
    _model: object = field(default=None, repr=False)
    _labels: list[str] = field(default_factory=list)

    def load(self) -> "SequenceClassifier":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        log(f"Loading {self.model_name} on {self.device}...")
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=autocast_dtype(self.device, self.fp32),
        )
        model.eval()
        model.to(self.device)
        self._model = model

        cfg = model.config
        order = sorted(cfg.id2label) if isinstance(cfg.id2label, dict) else range(cfg.num_labels)
        self._labels = [str(cfg.id2label[i]).lower() for i in order]
        log(f"  labels: {self._labels}")
        return self

    @property
    def labels(self) -> list[str]:
        return self._labels

    def _forward(self, texts: list[str]) -> list[np.ndarray]:
        import torch

        enc = self._tok(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            logits = self._model(**enc).logits.float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return list(probs)

    def predict_proba(self, texts: list[str], desc: str = "classify") -> np.ndarray:
        """
        Returns an (n_texts, n_labels) probability matrix in the input order.

        Texts are sorted by length internally so each batch pads to a similar
        width, then restored to the caller's order.
        """
        if not texts:
            return np.empty((0, len(self._labels)), dtype=np.float32)

        ordered, inverse = sort_by_length(texts)
        out = run_with_oom_backoff(self._forward, ordered, self.batch_size, desc=desc)
        return np.vstack(out)[inverse].astype(np.float32)


# ─── Optimized zero-shot NLI ────────────────────────────────────────────────────

@dataclass
class ZeroShotScorer:
    """
    NLI zero-shot classification over a fixed label set.

    Usage:
        z = ZeroShotScorer(labels={"qibla": "the qibla compass direction", ...},
                           device="cuda").load()
        hits = z.assign(sentences, threshold=0.75)
    """

    labels: dict[str, str]                    # aspect name -> natural-language description
    model_name: str = "facebook/bart-large-mnli"
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    top_k: int = 5                            # labels sent to NLI per sentence; 0 = all
    max_length: int = 200
    fp32: bool = False
    hypothesis_template: str = "This review is about {}."

    _tok: object = field(default=None, repr=False)
    _model: object = field(default=None, repr=False)
    _embedder: object = field(default=None, repr=False)
    _label_emb: np.ndarray | None = field(default=None, repr=False)
    _entail_idx: int = 2

    # ---- setup ----

    def load(self) -> "ZeroShotScorer":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        log(f"Loading NLI model {self.model_name} on {self.device}...")
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, torch_dtype=autocast_dtype(self.device, self.fp32),
        )
        model.eval().to(self.device)
        self._model = model

        # BART-MNLI orders labels contradiction/neutral/entailment; do not assume.
        id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        self._entail_idx = next(
            (i for i, v in id2label.items() if "entail" in v), len(id2label) - 1
        )
        log(f"  entailment logit index: {self._entail_idx} ({id2label})")

        if self.top_k and self.top_k < len(self.labels):
            self._load_prefilter()
        return self

    def _load_prefilter(self) -> None:
        from sentence_transformers import SentenceTransformer

        log(f"Loading prefilter encoder {self.embed_model_name} (top_k={self.top_k})...")
        self._embedder = SentenceTransformer(self.embed_model_name, device=self.device)
        descriptions = [self.labels[k] for k in self.label_names]
        self._label_emb = self._embedder.encode(
            descriptions, normalize_embeddings=True, show_progress_bar=False,
        )

    @property
    def label_names(self) -> list[str]:
        return list(self.labels.keys())

    # ---- scoring ----

    def _shortlist(self, sentences: list[str]) -> np.ndarray:
        """
        (n_sentences, top_k) matrix of candidate label indices.

        Cosine similarity between sentence and label-description embeddings.
        Cheap (MiniLM, one pass) relative to bart-large NLI.
        """
        if self._embedder is None or self._label_emb is None:
            n_labels = len(self.labels)
            return np.tile(np.arange(n_labels), (len(sentences), 1))

        emb = self._embedder.encode(
            sentences, batch_size=max(64, self.batch_size), normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True,
        )
        sims = emb @ self._label_emb.T                       # (n_sent, n_labels)
        return np.argpartition(-sims, kth=self.top_k - 1, axis=1)[:, : self.top_k]

    def _nli_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Entailment probability for each (premise, hypothesis) pair."""
        import torch

        premises = [p for p, _ in pairs]
        hypotheses = [h for _, h in pairs]
        enc = self._tok(
            premises, hypotheses, padding=True, truncation="only_first",
            max_length=self.max_length, return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            logits = self._model(**enc).logits.float()
            # Entailment vs contradiction, the standard zero-shot reduction.
            contra_idx = 0 if self._entail_idx != 0 else 2
            pair = torch.stack([logits[:, contra_idx], logits[:, self._entail_idx]], dim=-1)
            probs = torch.softmax(pair, dim=-1)[:, 1].cpu().numpy()
        return list(probs)

    def score(self, sentences: list[str]) -> np.ndarray:
        """
        (n_sentences, n_labels) entailment-probability matrix.

        Labels not shortlisted by the prefilter score 0.0 — they were never
        sent to the NLI model, which is the whole point of the prefilter.
        """
        names = self.label_names
        n, m = len(sentences), len(names)
        out = np.zeros((n, m), dtype=np.float32)
        if n == 0:
            return out

        # 1. Collapse exact duplicates — short review sentences repeat heavily.
        uniq, inverse = np.unique(np.asarray(sentences, dtype=object), return_inverse=True)
        uniq_list = [str(u) for u in uniq]
        log(f"Zero-shot: {n:,} sentences → {len(uniq_list):,} unique "
            f"({1 - len(uniq_list) / max(1, n):.0%} collapsed)")

        # 2. Shortlist candidate labels per unique sentence.
        cand = self._shortlist(uniq_list)
        log(f"Zero-shot: {len(uniq_list) * cand.shape[1]:,} NLI pairs "
            f"(vs {len(uniq_list) * m:,} without prefilter)")

        # 3. Build pairs, length-bucket, run with OOM backoff.
        pairs, coords = [], []
        for si, sent in enumerate(uniq_list):
            for li in cand[si]:
                pairs.append((sent, self.hypothesis_template.format(self.labels[names[li]])))
                coords.append((si, int(li)))

        order = np.argsort([len(p[0]) + len(p[1]) for p in pairs], kind="stable")
        ordered = [pairs[i] for i in order]
        scores = run_with_oom_backoff(self._nli_batch, ordered, self.batch_size, desc="nli")

        uniq_out = np.zeros((len(uniq_list), m), dtype=np.float32)
        for pos, sc in zip(order, scores):
            si, li = coords[pos]
            uniq_out[si, li] = sc

        # 4. Expand back to the original sentence order.
        out = uniq_out[inverse]
        return out

    def assign(self, sentences: list[str], threshold: float = 0.75) -> list[list[str]]:
        """Label names scoring above `threshold` for each sentence."""
        scores = self.score(sentences)
        names = np.array(self.label_names)
        return [list(names[row >= threshold]) for row in scores]

    def prefilter_recall(self, sentences: list[str], gold: list[list[str]]) -> float:
        """
        Fraction of gold labels the embedding prefilter keeps in its shortlist.

        Report this alongside any zero-shot result: it bounds the recall the
        NLI stage can possibly achieve, and reviewers will ask for it.
        """
        if self._embedder is None:
            return 1.0
        cand = self._shortlist(sentences)
        names = self.label_names
        kept = total = 0
        for row, g in zip(cand, gold):
            shortlisted = {names[i] for i in row}
            for label in g:
                total += 1
                kept += label in shortlisted
        return kept / total if total else 1.0
