"""
SmartTicket — Model-Agnostic Explainability Layer (LIME + SHAP)

Replaces the previous symbolic / rule-based explainer with two
post-hoc, model-agnostic explainers from the XAI literature:

  • LIME (Local Interpretable Model-agnostic Explanations)
        Ribeiro et al., 2016 — "Why Should I Trust You?"
        Perturbs the input text by randomly dropping words, queries
        the trained model, and fits a small *local* linear model on
        those perturbations.  The linear model's coefficients tell us
        which words pushed THIS prediction toward THIS class.

  • SHAP (SHapley Additive exPlanations)
        Lundberg & Lee, 2017 — game-theoretic feature attributions.
        Each feature's contribution = its average marginal effect over
        all possible feature subsets (Shapley value).  For sklearn
        pipelines we use shap.KernelExplainer over the TF-IDF feature
        space, which is fully model-agnostic.

Both explainers wrap a SmartTicket ML pipeline (TF-IDF vectorizer +
trained classifier) and produce ExplanationResult objects compatible
with the previous explainer's downstream consumers (Streamlit dashboard,
notebook, batch reports).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Callable

import numpy as np

# Optional heavy dependencies — imported lazily inside the methods that
# need them so that simply importing this module never crashes on a
# machine that hasn't installed lime/shap yet.


# ══════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════

@dataclass
class FeatureContribution:
    """A single feature's contribution to a prediction."""
    feature: str          # the word / token / feature name
    weight: float         # signed contribution (positive = pushes toward predicted class)
    abs_weight: float     # |weight| for ranking


@dataclass
class ExplanationResult:
    """Post-hoc explanation for one ticket classification."""

    # The prediction being explained
    predicted_label: str
    predicted_confidence: float

    # All class probabilities the model produced
    class_probabilities: Dict[str, float] = field(default_factory=dict)

    # LIME local-linear feature contributions (signed)
    lime_contributions: List[FeatureContribution] = field(default_factory=list)
    lime_intercept: float = 0.0
    lime_local_r2: float = 0.0   # how well the local linear approx fit the model

    # SHAP feature contributions (signed Shapley values)
    shap_contributions: List[FeatureContribution] = field(default_factory=list)
    shap_base_value: float = 0.0

    # Original input
    text: str = ""

    def top_words(self, k: int = 10, source: str = "lime") -> List[Tuple[str, float]]:
        """Return the top-k most influential words from the chosen explainer."""
        if source == "lime":
            contribs = self.lime_contributions
        elif source == "shap":
            contribs = self.shap_contributions
        else:
            raise ValueError("source must be 'lime' or 'shap'")
        return [(c.feature, c.weight) for c in contribs[:k]]

    def summary(self) -> str:
        """Concise human-readable summary."""
        lines = [
            "┌─ SmartTicket Explanation ────────────────────────────────┐",
            f"│  Predicted   : {self.predicted_label.upper():<20}"
            f"  conf={self.predicted_confidence:.0%}        │",
        ]
        if self.lime_contributions:
            lines.append("│  Top LIME words (local linear fit):                      │")
            for c in self.lime_contributions[:5]:
                sign = "+" if c.weight >= 0 else "-"
                line = f"│    {sign} {c.feature:<25} {abs(c.weight):>+.3f}"
                lines.append(f"{line:<60}│")
        if self.shap_contributions:
            lines.append("│  Top SHAP words (Shapley values):                        │")
            for c in self.shap_contributions[:5]:
                sign = "+" if c.weight >= 0 else "-"
                line = f"│    {sign} {c.feature:<25} {abs(c.weight):>+.3f}"
                lines.append(f"{line:<60}│")
        lines.append("└──────────────────────────────────────────────────────────┘")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# MLExplainer — LIME + SHAP wrapper around a trained pipeline
# ══════════════════════════════════════════════════════════════

class MLExplainer:
    """
    Model-agnostic explainer that combines LIME (text-domain perturbation)
    and SHAP (TF-IDF feature attribution) over a trained SmartTicket pipeline.

    Parameters
    ----------
    model : trained sklearn-compatible classifier
        Must expose ``predict_proba``.
    tfidf : fitted TfidfVectorizer
        Used both to transform raw text for SHAP and to recover feature names.
    class_names : list of str
        Ordered class labels (matches model.classes_ order).
    feature_builder : callable text -> sparse matrix, optional
        Function that turns raw text into the model's full feature row
        (TF-IDF + numeric + OHE).  If omitted, only TF-IDF is used and
        zeros are appended for the numeric / OHE columns when needed.
    n_lime_samples : int, default 1000
        Number of perturbations LIME generates per explanation.
    n_shap_background : int, default 30
        Background sample size for SHAP's KernelExplainer.
    """

    def __init__(
        self,
        model,
        tfidf,
        class_names: List[str],
        feature_builder: Optional[Callable] = None,
        n_lime_samples: int = 1000,
        n_shap_background: int = 30,
    ):
        self.model = model
        self.tfidf = tfidf
        self.class_names = list(class_names)
        self.feature_builder = feature_builder
        self.n_lime_samples = n_lime_samples
        self.n_shap_background = n_shap_background

        self._lime_explainer = None
        self._shap_explainer = None
        self._shap_background = None
        self._tfidf_feature_names = np.array(tfidf.get_feature_names_out())

    # ── Prediction adapters ──────────────────────────────────

    def _predict_proba_text(self, texts: List[str]) -> np.ndarray:
        """LIME calls this with a list of raw text strings."""
        if self.feature_builder is not None:
            X = self.feature_builder(texts)
        else:
            X = self.tfidf.transform(texts)
        return self.model.predict_proba(X)

    def _predict_proba_tfidf(self, X_tfidf) -> np.ndarray:
        """
        SHAP perturbs in TF-IDF space — but the trained model may expect a
        wider feature row (TF-IDF + numeric + OHE).  Pad with zero blocks
        of the right widths so the row matches what the model was fit on.
        """
        from scipy.sparse import csr_matrix, hstack, issparse

        if not issparse(X_tfidf):
            X_tfidf = csr_matrix(X_tfidf)

        n_tfidf = len(self._tfidf_feature_names)
        # If the caller passed a too-wide row, slice down to TF-IDF width.
        if X_tfidf.shape[1] > n_tfidf:
            X_tfidf = X_tfidf[:, :n_tfidf]

        # Pad to whatever the model was trained on (cached on first call).
        expected_n = self._expected_n_features()
        if expected_n == X_tfidf.shape[1]:
            return self.model.predict_proba(X_tfidf)

        n_pad = expected_n - X_tfidf.shape[1]
        if n_pad < 0:
            return self.model.predict_proba(X_tfidf[:, :expected_n])

        zeros = csr_matrix(np.zeros((X_tfidf.shape[0], n_pad)))
        return self.model.predict_proba(hstack([X_tfidf, zeros]))

    def _expected_n_features(self) -> int:
        """
        Infer the feature width the trained model expects.

        ``n_features_in_`` is set on most sklearn estimators after fit.
        For wrappers like our VotingEnsemble we fall back to probing with
        a single dummy text passed through ``feature_builder`` (if any).
        """
        n = getattr(self.model, "n_features_in_", None)
        if n is not None:
            return int(n)

        if self.feature_builder is not None:
            probe = self.feature_builder([" "])
            return int(probe.shape[1])

        return len(self._tfidf_feature_names)

    # ── Lazy initialisers ────────────────────────────────────

    def _get_lime(self):
        if self._lime_explainer is None:
            from lime.lime_text import LimeTextExplainer
            self._lime_explainer = LimeTextExplainer(
                class_names=self.class_names,
                bow=True,
                random_state=42,
            )
        return self._lime_explainer

    def _get_shap(self, background_texts: Optional[List[str]] = None):
        """
        Build a SHAP KernelExplainer.  Needs a small background dataset
        of TF-IDF rows to estimate "feature missing" baselines.
        """
        if self._shap_explainer is None:
            import shap

            if background_texts is None or len(background_texts) == 0:
                # Fallback: a single all-zero background row.
                background = np.zeros((1, len(self._tfidf_feature_names)))
            else:
                bg_n = min(self.n_shap_background, len(background_texts))
                background = self.tfidf.transform(background_texts[:bg_n]).toarray()

            self._shap_background = background
            self._shap_explainer = shap.KernelExplainer(
                self._predict_proba_tfidf,
                background,
            )
        return self._shap_explainer

    # ── Public API ───────────────────────────────────────────

    def explain(
        self,
        text: str,
        background_texts: Optional[List[str]] = None,
        num_features: int = 10,
        nsamples_shap: int = 100,
    ) -> ExplanationResult:
        """
        Run BOTH LIME and SHAP on a single ticket and return the combined result.
        """
        # 1. Predict
        proba = self._predict_proba_text([text])[0]
        pred_idx = int(np.argmax(proba))
        pred_label = self.class_names[pred_idx]
        class_probs = {c: float(p) for c, p in zip(self.class_names, proba)}

        # 2. LIME
        lime_contribs, lime_intercept, lime_r2 = self._lime_explain(
            text, pred_idx, num_features
        )

        # 3. SHAP
        shap_contribs, shap_base = self._shap_explain(
            text, pred_idx, num_features, background_texts, nsamples_shap
        )

        return ExplanationResult(
            predicted_label=pred_label,
            predicted_confidence=float(proba[pred_idx]),
            class_probabilities=class_probs,
            lime_contributions=lime_contribs,
            lime_intercept=lime_intercept,
            lime_local_r2=lime_r2,
            shap_contributions=shap_contribs,
            shap_base_value=shap_base,
            text=text,
        )

    def explain_lime(
        self,
        text: str,
        num_features: int = 10,
    ) -> ExplanationResult:
        """LIME-only explanation (cheaper than the combined one)."""
        proba = self._predict_proba_text([text])[0]
        pred_idx = int(np.argmax(proba))
        contribs, intercept, r2 = self._lime_explain(text, pred_idx, num_features)
        return ExplanationResult(
            predicted_label=self.class_names[pred_idx],
            predicted_confidence=float(proba[pred_idx]),
            class_probabilities={c: float(p) for c, p in zip(self.class_names, proba)},
            lime_contributions=contribs,
            lime_intercept=intercept,
            lime_local_r2=r2,
            text=text,
        )

    def explain_shap(
        self,
        text: str,
        background_texts: Optional[List[str]] = None,
        num_features: int = 10,
        nsamples: int = 100,
    ) -> ExplanationResult:
        """SHAP-only explanation."""
        proba = self._predict_proba_text([text])[0]
        pred_idx = int(np.argmax(proba))
        contribs, base = self._shap_explain(
            text, pred_idx, num_features, background_texts, nsamples
        )
        return ExplanationResult(
            predicted_label=self.class_names[pred_idx],
            predicted_confidence=float(proba[pred_idx]),
            class_probabilities={c: float(p) for c, p in zip(self.class_names, proba)},
            shap_contributions=contribs,
            shap_base_value=base,
            text=text,
        )

    def explain_batch(
        self,
        texts: List[str],
        background_texts: Optional[List[str]] = None,
        num_features: int = 10,
        nsamples_shap: int = 100,
    ) -> List[ExplanationResult]:
        """Apply explain() to a list of texts."""
        return [
            self.explain(t, background_texts, num_features, nsamples_shap)
            for t in texts
        ]

    # ── Internals ────────────────────────────────────────────

    def _lime_explain(
        self,
        text: str,
        target_class_idx: int,
        num_features: int,
    ) -> Tuple[List[FeatureContribution], float, float]:
        """Run LIME on a single text; return signed word contributions."""
        explainer = self._get_lime()
        exp = explainer.explain_instance(
            text_instance=text,
            classifier_fn=self._predict_proba_text,
            num_features=num_features,
            num_samples=self.n_lime_samples,
            labels=(target_class_idx,),
        )
        word_weights = exp.as_list(label=target_class_idx)
        contribs = [
            FeatureContribution(feature=w, weight=float(wt), abs_weight=abs(float(wt)))
            for w, wt in word_weights
        ]
        contribs.sort(key=lambda c: c.abs_weight, reverse=True)

        # Local linear-fit diagnostics (intercept + R²)
        try:
            intercept = float(exp.intercept[target_class_idx])
        except Exception:
            intercept = 0.0
        try:
            r2 = float(exp.score)
        except Exception:
            r2 = 0.0

        return contribs, intercept, r2

    def _shap_explain(
        self,
        text: str,
        target_class_idx: int,
        num_features: int,
        background_texts: Optional[List[str]],
        nsamples: int,
    ) -> Tuple[List[FeatureContribution], float]:
        """Run SHAP KernelExplainer on a single text; return signed Shapley values."""
        explainer = self._get_shap(background_texts)

        x_row = self.tfidf.transform([text]).toarray()
        shap_values = explainer.shap_values(
            x_row, nsamples=nsamples, silent=True
        )

        # KernelExplainer returns a list (one array per class) for
        # multi-class classifiers, or a single 2-D array.
        if isinstance(shap_values, list):
            class_shap = shap_values[target_class_idx][0]
            base_value = float(np.atleast_1d(explainer.expected_value)[target_class_idx])
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                class_shap = arr[0, :, target_class_idx]
            elif arr.ndim == 2:
                class_shap = arr[0]
            else:
                class_shap = arr
            base_value = float(np.atleast_1d(explainer.expected_value)[0])

        # Only report features that are actually present in the input
        # (TF-IDF entries with weight > 0).
        present_mask = x_row[0] > 0
        idxs = np.where(present_mask)[0]
        if len(idxs) == 0:
            return [], base_value

        contribs = [
            FeatureContribution(
                feature=str(self._tfidf_feature_names[i]),
                weight=float(class_shap[i]),
                abs_weight=abs(float(class_shap[i])),
            )
            for i in idxs
        ]
        contribs.sort(key=lambda c: c.abs_weight, reverse=True)
        return contribs[:num_features], base_value


# ══════════════════════════════════════════════════════════════
# Batch reporting
# ══════════════════════════════════════════════════════════════

def generate_batch_report(results: List[ExplanationResult]) -> str:
    """Aggregate statistics for a batch of explanations."""
    n = len(results)
    if n == 0:
        return "No results to report."

    # Aggregate top words across the batch (LIME + SHAP)
    lime_word_totals: Dict[str, float] = {}
    shap_word_totals: Dict[str, float] = {}
    for r in results:
        for c in r.lime_contributions:
            lime_word_totals[c.feature] = lime_word_totals.get(c.feature, 0.0) + c.abs_weight
        for c in r.shap_contributions:
            shap_word_totals[c.feature] = shap_word_totals.get(c.feature, 0.0) + c.abs_weight

    avg_conf = sum(r.predicted_confidence for r in results) / n
    avg_lime_r2 = sum(r.lime_local_r2 for r in results) / n

    pred_dist: Dict[str, int] = {}
    for r in results:
        pred_dist[r.predicted_label] = pred_dist.get(r.predicted_label, 0) + 1

    lines = [
        "=" * 62,
        "  SmartTicket — XAI Batch Report (LIME + SHAP)",
        "=" * 62,
        f"  Tickets explained        : {n}",
        f"  Avg model confidence     : {avg_conf:.0%}",
        f"  Avg LIME local R²        : {avg_lime_r2:.3f}",
        "",
        "  Predicted-class distribution:",
    ]
    for cls, count in sorted(pred_dist.items(), key=lambda kv: -kv[1]):
        bar_len = int(count / n * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        lines.append(f"    {cls:<14} {count:>4}  {bar}  {count/n:.0%}")

    if lime_word_totals:
        lines.append("")
        lines.append("  Top 10 globally influential words (LIME, summed |weight|):")
        for w, tot in sorted(lime_word_totals.items(), key=lambda kv: -kv[1])[:10]:
            lines.append(f"    {w:<25} {tot:.3f}")

    if shap_word_totals:
        lines.append("")
        lines.append("  Top 10 globally influential words (SHAP, summed |Shapley|):")
        for w, tot in sorted(shap_word_totals.items(), key=lambda kv: -kv[1])[:10]:
            lines.append(f"    {w:<25} {tot:.3f}")

    lines.append("=" * 62)
    return "\n".join(lines)


def print_xai_overview() -> None:
    """Print a short overview of how LIME and SHAP improve interpretability."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║      LIME + SHAP — Post-Hoc Model-Agnostic Explainability        ║
║      SmartTicket Explainability Layer                            ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  PROBLEM WITH PURE ML                                            ║
║  Voting / stacking ensembles built from KNN, RF, SVM, etc. are   ║
║  black boxes — they predict a department, but they cannot tell   ║
║  you *why*.  Stakeholders need word-level evidence; engineers    ║
║  need to debug failure modes; auditors need traceability.        ║
║                                                                  ║
║  1. LIME (LOCAL INTERPRETABLE MODEL-AGNOSTIC EXPLANATIONS)       ║
║     For each ticket, LIME perturbs the text by randomly          ║
║     dropping words, queries the model, and fits a tiny linear    ║
║     model in the neighbourhood of THIS prediction.  The linear   ║
║     coefficients tell us which words pushed the prediction       ║
║     toward the chosen class — and by how much.                   ║
║                                                                  ║
║  2. SHAP (SHAPLEY ADDITIVE EXPLANATIONS)                         ║
║     SHAP assigns each feature its average marginal contribution  ║
║     across all possible feature subsets — the unique allocation  ║
║     satisfying efficiency, symmetry, dummy, and additivity.      ║
║     For our pipeline we use KernelExplainer over TF-IDF, which   ║
║     is fully model-agnostic (works with KNN, RF, SVM, voting     ║
║     ensembles — anything with predict_proba).                    ║
║                                                                  ║
║  3. WHY USE BOTH                                                 ║
║     • LIME  : fast, intuitive, great for individual UI panels.   ║
║     • SHAP  : theoretically grounded, additive, comparable       ║
║       across tickets, supports global feature importance via     ║
║       summation.                                                 ║
║     Cross-checking the two improves trust: when LIME and SHAP    ║
║     agree on the top words, the explanation is robust.           ║
║                                                                  ║
║  4. MODEL-AGNOSTIC                                               ║
║     Neither method needs access to model internals — only        ║
║     predict_proba.  Swapping out the classifier requires no      ║
║     change to the explainer.                                     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")
