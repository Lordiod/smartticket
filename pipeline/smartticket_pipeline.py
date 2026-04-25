"""
SmartTicket — Data Engineering & Feature Selection Pipeline
Week 4 Deliverable

Reads raw support tickets from SQLite, preprocesses text/numeric/categorical
features, engineers derived features, and applies two feature selection
strategies (greedy forward heuristic + genetic algorithm) against a KNN
baseline classifier.
"""

# ══════════════════════════════════════════════════════════════
# SECTION 1 — Imports & Setup
# ══════════════════════════════════════════════════════════════

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
from sklearn.metrics import accuracy_score, classification_report

from voting_classifier import VotingEnsemble
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
random.seed(42)

print("=" * 60)
print("  SmartTicket — Data Engineering & Feature Selection")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
# SECTION 2 — Data Ingestion from SQLite
# ══════════════════════════════════════════════════════════════

print("\n[Step 1] Loading data from SQLite database...")

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database", "smartticket.db")
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

print(f"Loaded {len(df)} records")
print(f"Columns: {len(df.columns)}")
print(f"\nSample data:")
print(df.head())
print(f"\nMissing values:")
missing_report = df.isna().sum()
print(missing_report[missing_report > 0])
print(f"\nDepartment distribution:")
print(df["department"].value_counts())
print(f"\nPriority distribution:")
print(df["priority"].value_counts())

# Remove duplicates
before = len(df)
df = df.drop_duplicates(subset=["ticket_id"], keep="first")
print(f"\nRemoved {before - len(df)} duplicate tickets")
print(f"Records after dedup: {len(df)}")

# ── Class distribution chart ──
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"), exist_ok=True)
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

dept_counts = df["department"].value_counts()
axes[0].bar(dept_counts.index, dept_counts.values, color=sns.color_palette("viridis", len(dept_counts)))
axes[0].set_title("Department Distribution", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=30)
for i, v in enumerate(dept_counts.values):
    axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold", fontsize=10)

prio_counts = df["priority"].value_counts()
axes[1].bar(prio_counts.index, prio_counts.values, color=sns.color_palette("magma", len(prio_counts)))
axes[1].set_title("Priority Distribution", fontsize=14, fontweight="bold")
axes[1].set_ylabel("Count")
axes[1].tick_params(axis="x", rotation=30)
for i, v in enumerate(prio_counts.values):
    axes[1].text(i, v + 5, str(v), ha="center", fontweight="bold", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=150)
plt.close()
print("\nSaved class_distribution.png")

# ══════════════════════════════════════════════════════════════
# SECTION 3 — Text Preprocessing Pipeline
# ══════════════════════════════════════════════════════════════

# 3.1 Handle missing text
print("\n[Step 2] Preprocessing text data...")
missing = df["ticket_text"].isna().sum()
print(f"Missing ticket_text: {missing}")
df["ticket_text"] = df["ticket_text"].fillna("no description provided")

# 3.2 Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)             # URLs
    text = re.sub(r"\S+@\S+", "", text)              # emails
    text = re.sub(r"&\w+;", "", text)                # HTML entities
    text = re.sub(r"<[^>]+>", "", text)              # HTML tags
    text = re.sub(r"#\d+", "", text)                 # order numbers like #12345
    text = re.sub(r"\b\d+\.\d+\.\d+\.\d+\b", "", text)  # IP addresses
    text = re.sub(r"[^a-z\s]", "", text)             # non-alpha
    text = re.sub(r"\s+", " ", text).strip()         # normalize whitespace
    return text

df["clean_text"] = df["ticket_text"].apply(clean_text)

print("\nText cleaning — before vs after:")
for i in range(5):
    print(f"  BEFORE: {str(df['ticket_text'].iloc[i])[:80]}...")
    print(f"  AFTER : {df['clean_text'].iloc[i][:80]}...")
    print()

# 3.3 Remove stopwords
stop_words = set(stopwords.words("english"))
df["clean_text"] = df["clean_text"].apply(
    lambda x: " ".join([w for w in x.split() if w not in stop_words])
)

print("After stopword removal (5 samples):")
for i in range(5):
    print(f"  {df['clean_text'].iloc[i][:90]}")

# 3.4 Lemmatization
lemmatizer = WordNetLemmatizer()
df["clean_text"] = df["clean_text"].apply(
    lambda x: " ".join([lemmatizer.lemmatize(w) for w in x.split()])
)

print("\nAfter lemmatization (5 samples):")
for i in range(5):
    print(f"  {df['clean_text'].iloc[i][:90]}")

# 3.5 Text feature engineering
df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))
df["char_count"] = df["clean_text"].apply(lambda x: len(x))
df["avg_word_len"] = df.apply(
    lambda x: x["char_count"] / x["word_count"] if x["word_count"] > 0 else 0, axis=1
)
df["sentence_count"] = df["ticket_text"].apply(
    lambda x: str(x).count(".") + str(x).count("!") + str(x).count("?") + 1
)

print(f"\nText-derived features — word_count stats:")
print(df["word_count"].describe())

# 3.6 Remove text outliers
q_low = df["word_count"].quantile(0.01)
q_high = df["word_count"].quantile(0.99)
before = len(df)
df = df[(df["word_count"] >= q_low) & (df["word_count"] <= q_high)]
print(f"\nRemoved {before - len(df)} text outliers (word_count outside [{q_low}, {q_high}])")
print(f"Records remaining: {len(df)}")

# ══════════════════════════════════════════════════════════════
# SECTION 4 — Numeric Preprocessing
# ══════════════════════════════════════════════════════════════

print("\n[Step 3] Preprocessing numeric data...")

numeric_cols = [
    "account_age_days", "total_orders", "total_spent", "returns_count",
    "avg_order_value", "days_since_last_order", "previous_tickets",
    "avg_response_satisfaction", "response_time_hours", "num_attachments",
    "num_replies", "escalated", "reopened", "sentiment_score",
    "word_count_raw", "has_order_number",
]

# 4.1 Handle missing values
print("Missing values before fill:")
missing_before = df[numeric_cols].isna().sum()
print(missing_before[missing_before > 0])

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

print("Missing values after fill: 0")

# 4.2 Clip outliers
df["total_spent"] = df["total_spent"].clip(0, 50000)
df["account_age_days"] = df["account_age_days"].clip(0, 3650)
df["response_time_hours"] = df["response_time_hours"].clip(0, 168)
df["avg_response_satisfaction"] = df["avg_response_satisfaction"].clip(1, 5)
print("Outliers clipped to valid ranges")

# 4.3 Derived features
df["spend_per_order"] = df["total_spent"] / (df["total_orders"] + 1)
df["return_rate"] = df["returns_count"] / (df["total_orders"] + 1)
df["ticket_rate"] = df["previous_tickets"] / (df["account_age_days"] / 30 + 1)
df["order_recency_score"] = 1 / (df["days_since_last_order"] + 1)

derived_cols = ["spend_per_order", "return_rate", "ticket_rate", "order_recency_score"]
print(f"Engineered {len(derived_cols)} derived features: {derived_cols}")

# ══════════════════════════════════════════════════════════════
# SECTION 5 — Categorical Encoding
# ══════════════════════════════════════════════════════════════

print("\n[Step 4] Encoding categorical features...")

cat_cols = ["channel", "product_category", "region", "loyalty_tier"]

# 5.1 Standardize casing
for col in cat_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()
    df[col] = df[col].replace({"nan": "unknown", "none": "unknown", "": "unknown"})

print("Unique values after standardization:")
for col in cat_cols:
    print(f"  {col}: {sorted(df[col].unique())}")

# 5.2 One-hot encode
df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)
ohe_cols = [c for c in df.columns if any(c.startswith(cat + "_") for cat in cat_cols)]
print(f"One-hot encoded: {len(ohe_cols)} columns")

# 5.3 Encode targets
le_dept = LabelEncoder()
le_prio = LabelEncoder()
df["department_encoded"] = le_dept.fit_transform(df["department"])
df["priority_encoded"] = le_prio.fit_transform(df["priority"])
print(f"Department classes: {list(le_dept.classes_)}")
print(f"Priority classes: {list(le_prio.classes_)}")

# ══════════════════════════════════════════════════════════════
# SECTION 6 — Build Feature Matrix
# ══════════════════════════════════════════════════════════════

print("\n[Step 5] Building feature matrix...")

# TF-IDF
tfidf = TfidfVectorizer(max_features=500, min_df=2, max_df=0.95)
X_tfidf = tfidf.fit_transform(df["clean_text"])
print(f"TF-IDF shape: {X_tfidf.shape}")

# Scale numeric features
all_numeric = numeric_cols + derived_cols + ["word_count", "char_count", "avg_word_len", "sentence_count"]
all_numeric = [c for c in all_numeric if c in df.columns]

scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(df[all_numeric].values)
numeric_sparse = csr_matrix(numeric_scaled)

# OHE
ohe_sparse = csr_matrix(df[ohe_cols].values)

# Combine all
X_final = hstack([X_tfidf, numeric_sparse, ohe_sparse])
y_dept = df["department_encoded"].values

print(f"Final feature matrix: {X_final.shape}")

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X_final, y_dept, test_size=0.2, random_state=42, stratify=y_dept
)
print(f"Train: {X_train.shape[0]}, Validation: {X_val.shape[0]}")

# ══════════════════════════════════════════════════════════════
# SECTION 7 — Baseline KNN Model
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  BASELINE: KNN with ALL features")
print("=" * 60)

baseline = KNeighborsClassifier(n_neighbors=5)
baseline.fit(X_train, y_train)
baseline_preds = baseline.predict(X_val)
baseline_acc = accuracy_score(y_val, baseline_preds)
print(f"Accuracy: {baseline_acc:.4f} ({X_train.shape[1]} features)")
print(classification_report(y_val, baseline_preds, target_names=le_dept.classes_))

# ══════════════════════════════════════════════════════════════
# SECTION 8 — Feature Evaluation Function
# ══════════════════════════════════════════════════════════════

# Use only numeric + OHE features for feature selection (dense)
X_for_fs = np.hstack([numeric_scaled, df[ohe_cols].values.astype(float)])
y_for_fs = y_dept
feature_names_fs = all_numeric + ohe_cols
print(f"\nFeature selection matrix: {X_for_fs.shape}")
print(f"Feature names ({len(feature_names_fs)}): {feature_names_fs}")


def evaluate_features(X, y, feature_mask):
    """Train a quick KNN on selected features and return accuracy."""
    selected = X[:, feature_mask == 1]
    if selected.shape[1] == 0:
        return 0.0
    X_tr, X_te, y_tr, y_te = train_test_split(selected, y, test_size=0.3, random_state=42)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_tr, y_tr)
    return accuracy_score(y_te, model.predict(X_te))


# ══════════════════════════════════════════════════════════════
# SECTION 9 — Heuristic Feature Selection (Greedy Forward)
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  HEURISTIC: Greedy Forward Feature Selection")
print("=" * 60)


def heuristic_feature_selection(X, y, max_features=None):
    n_features = X.shape[1]
    if max_features is None:
        max_features = n_features
    selected_mask = np.zeros(n_features, dtype=int)
    best_score = 0

    print("Starting Heuristic Feature Selection...")
    for step in range(max_features):
        best_feature = -1
        best_temp_score = best_score
        for i in range(n_features):
            if selected_mask[i] == 1:
                continue
            temp_mask = selected_mask.copy()
            temp_mask[i] = 1
            score = evaluate_features(X, y, temp_mask)
            if score > best_temp_score:
                best_temp_score = score
                best_feature = i
        if best_feature == -1:
            print(f"  No improvement found at step {step+1}. Stopping early.")
            break
        selected_mask[best_feature] = 1
        best_score = best_temp_score
        print(f"  Step {step+1}: +{feature_names_fs[best_feature]:<30} Accuracy = {best_score:.4f}")

    return selected_mask, best_score


heuristic_mask, heuristic_score = heuristic_feature_selection(X_for_fs, y_for_fs, max_features=10)
heuristic_selected = [feature_names_fs[i] for i, v in enumerate(heuristic_mask) if v == 1]
print(f"\nHeuristic — Accuracy: {heuristic_score:.4f}, Features: {heuristic_selected}")

# ══════════════════════════════════════════════════════════════
# SECTION 10 — Genetic Algorithm Feature Selection
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  GENETIC ALGORITHM: Feature Selection")
print("=" * 60)


def initialize_population(pop_size, n_features):
    return np.array([np.random.randint(0, 2, n_features) for _ in range(pop_size)])


def select_parents(population, fitness_scores, num_parents):
    parents_idx = np.argsort(fitness_scores)[-num_parents:]
    return population[parents_idx]


def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    return np.concatenate([parent1[:point], parent2[point:]])


def mutate(individual, mutation_rate=0.05):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual


def genetic_feature_selection(X, y, pop_size=20, generations=15, mutation_rate=0.05):
    n_features = X.shape[1]
    population = initialize_population(pop_size, n_features)
    best_individual = None
    best_score = 0

    print("Starting Genetic Algorithm Feature Selection...")
    for gen in range(generations):
        fitness_scores = np.array([evaluate_features(X, y, ind) for ind in population])
        max_idx = np.argmax(fitness_scores)
        if fitness_scores[max_idx] > best_score:
            best_score = fitness_scores[max_idx]
            best_individual = population[max_idx].copy()
        print(f"  Generation {gen+1}/{generations}, Best Accuracy = {best_score:.4f}")

        parents = select_parents(population, fitness_scores, pop_size // 2)
        new_population = []
        while len(new_population) < pop_size:
            p1, p2 = random.sample(list(parents), 2)
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        population = np.array(new_population)

    return best_individual, best_score


ga_mask, ga_score = genetic_feature_selection(X_for_fs, y_for_fs)
ga_selected = [feature_names_fs[i] for i, v in enumerate(ga_mask) if v == 1]
print(f"\nGA — Accuracy: {ga_score:.4f}, Features: {ga_selected}")

# ══════════════════════════════════════════════════════════════
# SECTION 11 — Voting Ensemble Classification
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  VOTING ENSEMBLE: Multi-Algorithm Classification")
print("=" * 60)

# Define base estimators
base_estimators = [
    ("KNN", KNeighborsClassifier(n_neighbors=5)),
    ("DecisionTree", DecisionTreeClassifier(random_state=42, max_depth=15)),
    ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ("LogisticReg", LogisticRegression(max_iter=1000, random_state=42)),
    ("SVM", SVC(kernel="rbf", probability=True, random_state=42)),
]

# ── Hard voting ──
print("\n--- Hard Voting (majority vote) ---")
hard_ensemble = VotingEnsemble(estimators=base_estimators, voting="hard")
hard_ensemble.fit(X_train, y_train)

hard_preds = hard_ensemble.predict(X_val)
hard_acc = accuracy_score(y_val, hard_preds)
print(f"Hard Voting Accuracy: {hard_acc:.4f}")

# Individual model accuracies
print("\nIndividual model accuracies:")
individual_accs = hard_ensemble.get_individual_accuracies(X_val, y_val)
for name, acc in individual_accs.items():
    print(f"  {name:<20} {acc:.4f}")

print(f"\nHard Voting Classification Report:")
print(classification_report(y_val, hard_preds, target_names=le_dept.classes_))

# ── Soft voting ──
print("--- Soft Voting (averaged probabilities) ---")
soft_ensemble = VotingEnsemble(estimators=base_estimators, voting="soft")
soft_ensemble.fit(X_train, y_train)

soft_preds = soft_ensemble.predict(X_val)
soft_acc = accuracy_score(y_val, soft_preds)
print(f"Soft Voting Accuracy: {soft_acc:.4f}")

print(f"\nSoft Voting Classification Report:")
print(classification_report(y_val, soft_preds, target_names=le_dept.classes_))

# ── Weighted soft voting (weights based on individual performance) ──
print("--- Weighted Soft Voting ---")
# Use individual accuracies as weights so better classifiers count more
weight_values = [individual_accs[name] for name, _ in base_estimators]
print(f"Weights (from individual accuracy): {dict(zip([n for n,_ in base_estimators], [f'{w:.4f}' for w in weight_values]))}")

weighted_ensemble = VotingEnsemble(
    estimators=base_estimators, voting="soft", weights=weight_values
)
weighted_ensemble.fit(X_train, y_train)

weighted_preds = weighted_ensemble.predict(X_val)
weighted_acc = accuracy_score(y_val, weighted_preds)
print(f"Weighted Soft Voting Accuracy: {weighted_acc:.4f}")

print(f"\nWeighted Soft Voting Classification Report:")
print(classification_report(y_val, weighted_preds, target_names=le_dept.classes_))

# ══════════════════════════════════════════════════════════════
# SECTION 12 — Unified Pipeline Function
# ══════════════════════════════════════════════════════════════


def feature_selection_pipeline(X, y, method="heuristic"):
    """Run feature selection with the specified method."""
    if method == "heuristic":
        mask, score = heuristic_feature_selection(X, y, max_features=10)
    elif method == "ga":
        mask, score = genetic_feature_selection(X, y, pop_size=20, generations=15)
    else:
        raise ValueError("Method must be 'heuristic' or 'ga'")

    selected = X[:, mask == 1]
    print(f"\nFinal Accuracy: {score:.4f}")
    print(f"Selected Features: {np.sum(mask)}")
    return selected, mask, score


# ══════════════════════════════════════════════════════════════
# SECTION 12 — Comparison & Technical Note
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  COMPARISON: All Methods")
print("=" * 60)
print(f"{'Method':<30} {'Accuracy':<15} {'# Features'}")
print("-" * 65)
print(f"{'Baseline KNN (all+TF-IDF)':<30} {baseline_acc:<15.4f} {X_train.shape[1]}")
print(f"{'Heuristic (Greedy)':<30} {heuristic_score:<15.4f} {sum(heuristic_mask)}")
print(f"{'Genetic Algorithm':<30} {ga_score:<15.4f} {sum(ga_mask)}")
print(f"{'Hard Voting Ensemble':<30} {hard_acc:<15.4f} {X_train.shape[1]}")
print(f"{'Soft Voting Ensemble':<30} {soft_acc:<15.4f} {X_train.shape[1]}")
print(f"{'Weighted Soft Voting':<30} {weighted_acc:<15.4f} {X_train.shape[1]}")

# ── Comparison chart ──
methods = [
    "Baseline\nKNN", "Heuristic\n(Greedy)", "Genetic\nAlgorithm",
    "Hard\nVoting", "Soft\nVoting", "Weighted\nSoft Voting",
]
accuracies = [baseline_acc, heuristic_score, ga_score, hard_acc, soft_acc, weighted_acc]
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(methods, accuracies, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{acc:.4f}", ha="center", fontweight="bold", fontsize=11)
ax.set_ylim(0, max(accuracies) * 1.15)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("SmartTicket — Method Accuracy Comparison", fontsize=14, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "comparison_chart.png"), dpi=150)
plt.close()
print("\nSaved comparison_chart.png")

# ── Technical note ──
print("""
╔══════════════════════════════════════════════════════════╗
║        Feature Selection & Ensemble Strategies           ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Heuristic (Greedy Forward):                             ║
║    Adds one feature at a time, picking whichever gives   ║
║    the best accuracy boost. Deterministic. Fast.         ║
║    May miss beneficial feature interactions.             ║
║                                                          ║
║  Genetic Algorithm:                                      ║
║    Searches over entire feature subsets using selection,  ║
║    crossover, and mutation. Stochastic. Slower but can   ║
║    discover non-obvious feature combinations.            ║
║                                                          ║
║  Voting Ensemble:                                        ║
║    Combines KNN, Decision Tree, Random Forest,           ║
║    Logistic Regression, and SVM into a single            ║
║    prediction via majority vote (hard) or averaged       ║
║    probabilities (soft). Weighted soft voting uses       ║
║    individual accuracy to emphasize better models.       ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

print("=" * 60)
print("  Pipeline Complete!")
print("=" * 60)
