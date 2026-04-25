"""
SmartTicket — Core Pipeline Functions (importable)

Refactored from smartticket_pipeline.py so the data processing,
feature engineering, and model training steps can be reused by
the Streamlit dashboard without re-running the entire script.
"""

import sqlite3
import os
import re
import random
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.sparse import hstack, csr_matrix

import ssl
import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from voting_classifier import VotingEnsemble

np.random.seed(42)
random.seed(42)

# ── Constants ────────────────────────────────────────────────

NUMERIC_COLS = [
    "account_age_days", "total_orders", "total_spent", "returns_count",
    "avg_order_value", "days_since_last_order", "previous_tickets",
    "avg_response_satisfaction", "response_time_hours", "num_attachments",
    "num_replies", "escalated", "reopened", "sentiment_score",
    "word_count_raw", "has_order_number",
]

CAT_COLS = ["channel", "product_category", "region", "loyalty_tier"]

DERIVED_COLS = ["spend_per_order", "return_rate", "ticket_rate", "order_recency_score"]

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ══════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════

def load_raw_data(db_path=None):
    """Load raw data from SQLite, return DataFrame before any processing."""
    if db_path is None:
        db_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "database", "smartticket.db"
        )
    conn = sqlite3.connect(db_path)
    query = """
    SELECT
        t.ticket_id, t.ticket_text, t.channel, t.product_category,
        t.region, t.department, t.priority,
        c.account_age_days, c.total_orders, c.total_spent,
        c.returns_count, c.avg_order_value, c.days_since_last_order,
        c.loyalty_tier, c.previous_tickets, c.avg_response_satisfaction,
        m.response_time_hours, m.num_attachments, m.num_replies,
        m.escalated, m.reopened, m.sentiment_score,
        m.word_count_raw, m.has_order_number
    FROM tickets t
    LEFT JOIN customer_metrics c ON t.ticket_id = c.ticket_id
    LEFT JOIN ticket_metadata m ON t.ticket_id = m.ticket_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# ══════════════════════════════════════════════════════════════
# Text Preprocessing
# ══════════════════════════════════════════════════════════════

def clean_text(text):
    """Clean a single text string."""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"&\w+;", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"#\d+", "", text)
    text = re.sub(r"\b\d+\.\d+\.\d+\.\d+\b", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(df):
    """Full text pipeline: fill missing, clean, stopwords, lemmatize, derive features."""
    df = df.copy()
    df["ticket_text"] = df["ticket_text"].fillna("no description provided")
    df["clean_text"] = df["ticket_text"].apply(clean_text)
    df["clean_text"] = df["clean_text"].apply(
        lambda x: " ".join([w for w in x.split() if w not in STOP_WORDS])
    )
    df["clean_text"] = df["clean_text"].apply(
        lambda x: " ".join([LEMMATIZER.lemmatize(w) for w in x.split()])
    )
    df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))
    df["char_count"] = df["clean_text"].apply(lambda x: len(x))
    df["avg_word_len"] = df.apply(
        lambda x: x["char_count"] / x["word_count"] if x["word_count"] > 0 else 0, axis=1
    )
    df["sentence_count"] = df["ticket_text"].apply(
        lambda x: str(x).count(".") + str(x).count("!") + str(x).count("?") + 1
    )
    # Remove text outliers
    q_low = df["word_count"].quantile(0.01)
    q_high = df["word_count"].quantile(0.99)
    df = df[(df["word_count"] >= q_low) & (df["word_count"] <= q_high)]
    return df


# ══════════════════════════════════════════════════════════════
# Numeric & Categorical Preprocessing
# ══════════════════════════════════════════════════════════════

def preprocess_numeric(df):
    """Handle missing values, clip outliers, derive features."""
    df = df.copy()
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df["total_spent"] = df["total_spent"].clip(0, 50000)
    df["account_age_days"] = df["account_age_days"].clip(0, 3650)
    df["response_time_hours"] = df["response_time_hours"].clip(0, 168)
    df["avg_response_satisfaction"] = df["avg_response_satisfaction"].clip(1, 5)

    df["spend_per_order"] = df["total_spent"] / (df["total_orders"] + 1)
    df["return_rate"] = df["returns_count"] / (df["total_orders"] + 1)
    df["ticket_rate"] = df["previous_tickets"] / (df["account_age_days"] / 30 + 1)
    df["order_recency_score"] = 1 / (df["days_since_last_order"] + 1)
    return df


def preprocess_categorical(df):
    """Standardize casing, fill unknowns, one-hot encode."""
    df = df.copy()
    for col in CAT_COLS:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].replace({"nan": "unknown", "none": "unknown", "": "unknown"})
    df = pd.get_dummies(df, columns=CAT_COLS, drop_first=True, dtype=float)
    return df


# ══════════════════════════════════════════════════════════════
# Full Pipeline
# ══════════════════════════════════════════════════════════════

def run_full_pipeline(db_path=None):
    """
    Run the entire preprocessing pipeline and return everything
    needed for modeling and visualization.
    """
    # Load
    df_raw = load_raw_data(db_path)
    before_dedup = len(df_raw)
    df = df_raw.drop_duplicates(subset=["ticket_id"], keep="first")
    n_dupes = before_dedup - len(df)

    # Preprocess
    df = preprocess_text(df)
    df = preprocess_numeric(df)
    df = preprocess_categorical(df)

    # Identify OHE columns
    ohe_cols = [c for c in df.columns if any(c.startswith(cat + "_") for cat in CAT_COLS)]

    # Encode targets
    le_dept = LabelEncoder()
    le_prio = LabelEncoder()
    df["department_encoded"] = le_dept.fit_transform(df["department"])
    df["priority_encoded"] = le_prio.fit_transform(df["priority"])

    # Build feature matrix
    tfidf = TfidfVectorizer(max_features=500, min_df=2, max_df=0.95)
    X_tfidf = tfidf.fit_transform(df["clean_text"])

    all_numeric = NUMERIC_COLS + DERIVED_COLS + ["word_count", "char_count", "avg_word_len", "sentence_count"]
    all_numeric = [c for c in all_numeric if c in df.columns]

    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(df[all_numeric].values)
    numeric_sparse = csr_matrix(numeric_scaled)
    ohe_sparse = csr_matrix(df[ohe_cols].values)

    X_final = hstack([X_tfidf, numeric_sparse, ohe_sparse])
    y_dept = df["department_encoded"].values

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_final, y_dept, test_size=0.2, random_state=42, stratify=y_dept
    )

    return {
        "df_raw": df_raw,
        "df": df,
        "n_dupes": n_dupes,
        "le_dept": le_dept,
        "le_prio": le_prio,
        "tfidf": tfidf,
        "scaler": scaler,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "all_numeric": all_numeric,
        "ohe_cols": ohe_cols,
        "numeric_scaled": numeric_scaled,
        "feature_names_fs": all_numeric + ohe_cols,
    }


# ══════════════════════════════════════════════════════════════
# Model Training
# ══════════════════════════════════════════════════════════════

def get_base_estimators():
    """Return the standard set of base estimators."""
    return [
        ("KNN", KNeighborsClassifier(n_neighbors=5)),
        ("Decision Tree", DecisionTreeClassifier(random_state=42, max_depth=15)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("SVM", SVC(kernel="rbf", probability=True, random_state=42)),
    ]


def train_all_models(X_train, X_val, y_train, y_val, le_dept):
    """Train baseline + all voting ensembles, return results dict."""
    base_estimators = get_base_estimators()

    # Baseline KNN
    baseline = KNeighborsClassifier(n_neighbors=5)
    baseline.fit(X_train, y_train)
    baseline_preds = baseline.predict(X_val)
    baseline_acc = accuracy_score(y_val, baseline_preds)

    # Hard voting
    hard_ens = VotingEnsemble(estimators=base_estimators, voting="hard")
    hard_ens.fit(X_train, y_train)
    hard_preds = hard_ens.predict(X_val)
    hard_acc = accuracy_score(y_val, hard_preds)
    individual_accs = hard_ens.get_individual_accuracies(X_val, y_val)

    # Soft voting
    soft_ens = VotingEnsemble(estimators=base_estimators, voting="soft")
    soft_ens.fit(X_train, y_train)
    soft_preds = soft_ens.predict(X_val)
    soft_acc = accuracy_score(y_val, soft_preds)

    # Weighted soft voting
    weight_values = [individual_accs[name] for name, _ in base_estimators]
    weighted_ens = VotingEnsemble(estimators=base_estimators, voting="soft", weights=weight_values)
    weighted_ens.fit(X_train, y_train)
    weighted_preds = weighted_ens.predict(X_val)
    weighted_acc = accuracy_score(y_val, weighted_preds)

    classes = le_dept.classes_

    return {
        "baseline": {
            "accuracy": baseline_acc,
            "predictions": baseline_preds,
            "report": classification_report(y_val, baseline_preds, target_names=classes, output_dict=True),
            "confusion": confusion_matrix(y_val, baseline_preds),
            "model": baseline,
        },
        "hard_voting": {
            "accuracy": hard_acc,
            "predictions": hard_preds,
            "report": classification_report(y_val, hard_preds, target_names=classes, output_dict=True),
            "confusion": confusion_matrix(y_val, hard_preds),
            "model": hard_ens,
        },
        "soft_voting": {
            "accuracy": soft_acc,
            "predictions": soft_preds,
            "report": classification_report(y_val, soft_preds, target_names=classes, output_dict=True),
            "confusion": confusion_matrix(y_val, soft_preds),
            "model": soft_ens,
        },
        "weighted_voting": {
            "accuracy": weighted_acc,
            "predictions": weighted_preds,
            "report": classification_report(y_val, weighted_preds, target_names=classes, output_dict=True),
            "confusion": confusion_matrix(y_val, weighted_preds),
            "model": weighted_ens,
        },
        "individual_accuracies": individual_accs,
        "weights": dict(zip([n for n, _ in base_estimators], weight_values)),
        "classes": classes,
    }


def predict_single_ticket(text, pipeline_data, models_result):
    """Classify a single ticket text through the full pipeline."""
    cleaned = clean_text(text)
    cleaned = " ".join([w for w in cleaned.split() if w not in STOP_WORDS])
    cleaned = " ".join([LEMMATIZER.lemmatize(w) for w in cleaned.split()])

    X_tfidf = pipeline_data["tfidf"].transform([cleaned])

    # For numeric/categorical, use median values (we don't have them for a raw text)
    n_numeric = len(pipeline_data["all_numeric"])
    n_ohe = len(pipeline_data["ohe_cols"])
    numeric_zeros = csr_matrix(np.zeros((1, n_numeric)))
    ohe_zeros = csr_matrix(np.zeros((1, n_ohe)))

    X_single = hstack([X_tfidf, numeric_zeros, ohe_zeros])

    le = pipeline_data["le_dept"]
    results = {}
    for name, data in models_result.items():
        if name in ("individual_accuracies", "weights", "classes"):
            continue
        model = data["model"]
        pred = model.predict(X_single)
        label = le.inverse_transform(pred)[0]
        results[name] = label

    return results
