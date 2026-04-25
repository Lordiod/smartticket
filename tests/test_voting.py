"""
Tests for the VotingEnsemble classifier.

Covers: hard voting, soft voting, weighted voting, edge cases,
        diagnostics, and validation.
"""

import sys
import os
import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Allow imports from pipeline/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from voting_classifier import VotingEnsemble


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def binary_data():
    """Simple binary classification dataset."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_classes=2, random_state=42,
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)


@pytest.fixture
def multiclass_data():
    """Six-class dataset matching SmartTicket's department count."""
    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=12,
        n_classes=6, n_clusters_per_class=1, random_state=42,
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def simple_estimators():
    """Minimal pair of estimators for quick tests."""
    return [
        ("KNN", KNeighborsClassifier(n_neighbors=3)),
        ("DT", DecisionTreeClassifier(random_state=42)),
    ]


@pytest.fixture
def full_estimators():
    """Full set of estimators matching the pipeline."""
    return [
        ("KNN", KNeighborsClassifier(n_neighbors=5)),
        ("DT", DecisionTreeClassifier(random_state=42, max_depth=15)),
        ("RF", RandomForestClassifier(n_estimators=50, random_state=42)),
        ("LR", LogisticRegression(max_iter=500, random_state=42)),
        ("SVM", SVC(kernel="rbf", probability=True, random_state=42)),
    ]


# ══════════════════════════════════════════════════════════════
# Hard Voting Tests
# ══════════════════════════════════════════════════════════════

class TestHardVoting:

    def test_basic_predict(self, binary_data, simple_estimators):
        X_train, X_test, y_train, y_test = binary_data
        ens = VotingEnsemble(simple_estimators, voting="hard")
        ens.fit(X_train, y_train)
        preds = ens.predict(X_test)

        assert preds.shape == y_test.shape
        assert set(preds).issubset(set(y_train))

    def test_accuracy_above_random(self, binary_data, simple_estimators):
        X_train, X_test, y_train, y_test = binary_data
        ens = VotingEnsemble(simple_estimators, voting="hard")
        ens.fit(X_train, y_train)
        acc = accuracy_score(y_test, ens.predict(X_test))

        # Ensemble should beat random (50%) on a well-separated dataset
        assert acc > 0.6

    def test_multiclass(self, multiclass_data, full_estimators):
        X_train, X_test, y_train, y_test = multiclass_data
        ens = VotingEnsemble(full_estimators, voting="hard")
        ens.fit(X_train, y_train)
        preds = ens.predict(X_test)

        assert preds.shape == y_test.shape
        assert set(preds).issubset(set(y_train))
        acc = accuracy_score(y_test, preds)
        # Should beat random (1/6 ≈ 16.7%) by a solid margin
        assert acc > 0.4

    def test_hard_voting_no_predict_proba(self, binary_data, simple_estimators):
        X_train, X_test, y_train, y_test = binary_data
        ens = VotingEnsemble(simple_estimators, voting="hard")
        ens.fit(X_train, y_train)

        with pytest.raises(AttributeError, match="predict_proba"):
            ens.predict_proba(X_test)


# ══════════════════════════════════════════════════════════════
# Soft Voting Tests
# ══════════════════════════════════════════════════════════════

class TestSoftVoting:

    def test_basic_predict(self, binary_data, full_estimators):
        X_train, X_test, y_train, y_test = binary_data
        ens = VotingEnsemble(full_estimators, voting="soft")
        ens.fit(X_train, y_train)
        preds = ens.predict(X_test)

        assert preds.shape == y_test.shape

    def test_predict_proba_shape(self, binary_data, full_estimators):
        X_train, X_test, y_train, y_test = binary_data
        ens = VotingEnsemble(full_estimators, voting="soft")
        ens.fit(X_train, y_train)
        probas = ens.predict_proba(X_test)

        assert probas.shape == (X_test.shape[0], 2)

    def test_probas_sum_to_one(self, binary_data, full_estimators):
        X_train, X_test, y_train, y_test = binary_data
        ens = VotingEnsemble(full_estimators, voting="soft")
        ens.fit(X_train, y_train)
        probas = ens.predict_proba(X_test)

        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_multiclass_probas(self, multiclass_data, full_estimators):
        X_train, X_test, y_train, y_test = multiclass_data
        ens = VotingEnsemble(full_estimators, voting="soft")
        ens.fit(X_train, y_train)
        probas = ens.predict_proba(X_test)

        assert probas.shape == (X_test.shape[0], 6)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_soft_beats_random(self, multiclass_data, full_estimators):
        X_train, X_test, y_train, y_test = multiclass_data
        ens = VotingEnsemble(full_estimators, voting="soft")
        ens.fit(X_train, y_train)
        acc = accuracy_score(y_test, ens.predict(X_test))
        assert acc > 0.4


# ══════════════════════════════════════════════════════════════
# Weighted Voting Tests
# ══════════════════════════════════════════════════════════════

class TestWeightedVoting:

    def test_weighted_hard_voting(self, binary_data, simple_estimators):
        X_train, X_test, y_train, y_test = binary_data
        weights = [2.0, 1.0]  # KNN counts double
        ens = VotingEnsemble(simple_estimators, voting="hard", weights=weights)
        ens.fit(X_train, y_train)
        preds = ens.predict(X_test)

        assert preds.shape == y_test.shape

    def test_weighted_soft_voting(self, binary_data, full_estimators):
        X_train, X_test, y_train, y_test = binary_data
        weights = [1.0, 0.8, 1.5, 1.2, 1.0]
        ens = VotingEnsemble(full_estimators, voting="soft", weights=weights)
        ens.fit(X_train, y_train)
        preds = ens.predict(X_test)

        assert preds.shape == y_test.shape

    def test_heavy_weight_dominates(self, binary_data):
        """When one estimator has overwhelming weight, it should dominate."""
        X_train, X_test, y_train, y_test = binary_data
        estimators = [
            ("KNN", KNeighborsClassifier(n_neighbors=3)),
            ("DT", DecisionTreeClassifier(random_state=42)),
        ]
        # Give KNN an enormous weight
        ens = VotingEnsemble(estimators, voting="hard", weights=[1000.0, 0.001])
        ens.fit(X_train, y_train)

        ens_preds = ens.predict(X_test)
        knn_preds = ens.fitted_estimators_[0][1].predict(X_test)

        # Ensemble should match KNN almost exactly
        np.testing.assert_array_equal(ens_preds, knn_preds)

    def test_equal_weights_matches_unweighted(self, binary_data, simple_estimators):
        X_train, X_test, y_train, y_test = binary_data

        unweighted = VotingEnsemble(simple_estimators, voting="soft")
        unweighted.fit(X_train, y_train)

        weighted = VotingEnsemble(simple_estimators, voting="soft", weights=[1.0, 1.0])
        weighted.fit(X_train, y_train)

        np.testing.assert_array_equal(
            unweighted.predict(X_test),
            weighted.predict(X_test),
        )


# ══════════════════════════════════════════════════════════════
# Diagnostics Tests
# ══════════════════════════════════════════════════════════════

class TestDiagnostics:

    def test_individual_predictions(self, binary_data, simple_estimators):
        X_train, X_test, y_train, y_test = binary_data
        ens = VotingEnsemble(simple_estimators, voting="hard")
        ens.fit(X_train, y_train)
        indiv = ens.get_individual_predictions(X_test)

        assert set(indiv.keys()) == {"KNN", "DT"}
        for preds in indiv.values():
            assert preds.shape == y_test.shape

    def test_individual_accuracies(self, binary_data, simple_estimators):
        X_train, X_test, y_train, y_test = binary_data
        ens = VotingEnsemble(simple_estimators, voting="hard")
        ens.fit(X_train, y_train)
        accs = ens.get_individual_accuracies(X_test, y_test)

        assert set(accs.keys()) == {"KNN", "DT"}
        for acc in accs.values():
            assert 0.0 <= acc <= 1.0

    def test_repr(self, simple_estimators):
        ens = VotingEnsemble(simple_estimators, voting="soft", weights=[1.0, 2.0])
        r = repr(ens)
        assert "KNN" in r
        assert "DT" in r
        assert "soft" in r


# ══════════════════════════════════════════════════════════════
# Validation & Edge Case Tests
# ══════════════════════════════════════════════════════════════

class TestValidation:

    def test_invalid_voting_mode(self):
        estimators = [
            ("A", KNeighborsClassifier()),
            ("B", DecisionTreeClassifier()),
        ]
        with pytest.raises(ValueError, match="voting must be"):
            VotingEnsemble(estimators, voting="invalid")

    def test_single_estimator_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            VotingEnsemble([("A", KNeighborsClassifier())], voting="hard")

    def test_wrong_weights_length(self):
        estimators = [
            ("A", KNeighborsClassifier()),
            ("B", DecisionTreeClassifier()),
        ]
        with pytest.raises(ValueError, match="weights"):
            VotingEnsemble(estimators, voting="hard", weights=[1.0])

    def test_predict_before_fit_raises(self):
        estimators = [
            ("A", KNeighborsClassifier()),
            ("B", DecisionTreeClassifier()),
        ]
        ens = VotingEnsemble(estimators, voting="hard")
        X = np.array([[1, 2], [3, 4]])
        with pytest.raises(RuntimeError, match="not been fitted"):
            ens.predict(X)

    def test_fitted_estimators_are_clones(self, binary_data, simple_estimators):
        """Ensure base estimators are cloned, not mutated."""
        X_train, X_test, y_train, y_test = binary_data
        original_knn = simple_estimators[0][1]

        ens = VotingEnsemble(simple_estimators, voting="hard")
        ens.fit(X_train, y_train)

        # Original estimator should NOT be fitted
        assert not hasattr(original_knn, "classes_")


# ══════════════════════════════════════════════════════════════
# Comparative Tests (soft vs hard)
# ══════════════════════════════════════════════════════════════

class TestComparison:

    def test_hard_and_soft_produce_valid_predictions(self, multiclass_data, full_estimators):
        X_train, X_test, y_train, y_test = multiclass_data

        hard = VotingEnsemble(full_estimators, voting="hard")
        hard.fit(X_train, y_train)
        hard_preds = hard.predict(X_test)

        soft = VotingEnsemble(full_estimators, voting="soft")
        soft.fit(X_train, y_train)
        soft_preds = soft.predict(X_test)

        # Both should produce valid class labels
        classes = set(y_train)
        assert set(hard_preds).issubset(classes)
        assert set(soft_preds).issubset(classes)

    def test_ensemble_at_least_as_good_as_worst_member(self, multiclass_data, full_estimators):
        """Ensemble should generally not be worse than the worst individual."""
        X_train, X_test, y_train, y_test = multiclass_data

        ens = VotingEnsemble(full_estimators, voting="soft")
        ens.fit(X_train, y_train)
        ens_acc = accuracy_score(y_test, ens.predict(X_test))

        individual_accs = ens.get_individual_accuracies(X_test, y_test)
        worst_acc = min(individual_accs.values())

        # Ensemble should beat or match the worst individual model
        assert ens_acc >= worst_acc - 0.05  # small tolerance for stochasticity
