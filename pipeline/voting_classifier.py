"""
SmartTicket — Voting Ensemble Classifier
Week 5 Deliverable

Custom voting ensemble that combines multiple classifiers using
hard voting (majority vote), soft voting (averaged probabilities),
or weighted variants of each.
"""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score


class VotingEnsemble:
    """
    Voting ensemble that combines predictions from multiple classifiers.

    Parameters
    ----------
    estimators : list of (name, estimator) tuples
        The base classifiers to combine.
    voting : str, default="hard"
        "hard" — majority vote on predicted labels.
        "soft" — average predicted probabilities, then pick argmax.
    weights : list of float or None, default=None
        Per-estimator weights.  When None every estimator counts equally.
    """

    VALID_VOTING = ("hard", "soft")

    def __init__(self, estimators, voting="hard", weights=None):
        if voting not in self.VALID_VOTING:
            raise ValueError(f"voting must be one of {self.VALID_VOTING}, got '{voting}'")
        if len(estimators) < 2:
            raise ValueError("VotingEnsemble requires at least 2 estimators")
        if weights is not None and len(weights) != len(estimators):
            raise ValueError("Length of weights must match number of estimators")

        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.fitted_estimators_ = []
        self.classes_ = None
        self.n_classes_ = 0

    # ── Fit ──────────────────────────────────────────────────

    def fit(self, X, y):
        """Fit every base estimator on the same training data."""
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.fitted_estimators_ = []

        for name, est in self.estimators:
            fitted = clone(est).fit(X, y)
            self.fitted_estimators_.append((name, fitted))

        return self

    # ── Predict ──────────────────────────────────────────────

    def predict(self, X):
        """Return class predictions using the chosen voting strategy."""
        self._check_is_fitted()

        if self.voting == "hard":
            return self._hard_vote(X)
        return self._soft_vote(X)

    def predict_proba(self, X):
        """Return averaged class probabilities (soft voting only)."""
        if self.voting == "hard":
            raise AttributeError(
                "predict_proba is only available when voting='soft'"
            )
        self._check_is_fitted()
        return self._collect_probas(X)

    # ── Hard voting ──────────────────────────────────────────

    def _hard_vote(self, X):
        """Majority vote: each estimator casts one vote per sample."""
        predictions = np.array(
            [est.predict(X) for _, est in self.fitted_estimators_]
        )  # shape: (n_estimators, n_samples)

        weights = self._normalised_weights()

        n_samples = predictions.shape[1]
        result = np.empty(n_samples, dtype=self.classes_.dtype)

        for i in range(n_samples):
            votes = np.zeros(self.n_classes_)
            for j, pred in enumerate(predictions[:, i]):
                class_idx = np.searchsorted(self.classes_, pred)
                votes[class_idx] += weights[j]
            result[i] = self.classes_[np.argmax(votes)]

        return result

    # ── Soft voting ──────────────────────────────────────────

    def _soft_vote(self, X):
        """Average predicted probabilities, then pick the argmax class."""
        avg_probas = self._collect_probas(X)
        return self.classes_[np.argmax(avg_probas, axis=1)]

    def _collect_probas(self, X):
        """Weighted average of predict_proba outputs."""
        weights = self._normalised_weights()

        probas = np.array(
            [est.predict_proba(X) for _, est in self.fitted_estimators_]
        )  # shape: (n_estimators, n_samples, n_classes)

        weighted = np.tensordot(weights, probas, axes=([0], [0]))
        return weighted  # shape: (n_samples, n_classes)

    # ── Diagnostics ──────────────────────────────────────────

    def get_individual_predictions(self, X):
        """Return each estimator's predictions as a dict."""
        self._check_is_fitted()
        return {name: est.predict(X) for name, est in self.fitted_estimators_}

    def get_individual_accuracies(self, X, y):
        """Return each estimator's accuracy as a dict."""
        self._check_is_fitted()
        return {
            name: accuracy_score(y, est.predict(X))
            for name, est in self.fitted_estimators_
        }

    # ── Helpers ───────────────────────────────────────────────

    def _normalised_weights(self):
        """Return weights that sum to 1."""
        if self.weights is None:
            n = len(self.fitted_estimators_)
            return np.ones(n) / n
        w = np.asarray(self.weights, dtype=float)
        return w / w.sum()

    def _check_is_fitted(self):
        if not self.fitted_estimators_:
            raise RuntimeError("VotingEnsemble has not been fitted yet")

    def __repr__(self):
        names = [name for name, _ in self.estimators]
        return (
            f"VotingEnsemble(estimators={names}, voting='{self.voting}', "
            f"weights={self.weights})"
        )
