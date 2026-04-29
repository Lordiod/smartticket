# SmartTicket — AI-Powered Support Ticket Classification

## Project Proposal

---

### 1. Problem Statement and Use Case

Customer support teams at mid-to-large companies receive hundreds — often thousands — of support tickets daily across email, chat, phone, social media, and web forms. Each ticket must be routed to the correct **department** (billing, technical, shipping, account, returns, or general) and assigned a **priority** level (low, medium, high, urgent). Today this triage is done manually, leading to:

- **Slow response times** — tickets sit in a general queue until a human reads and re-routes them.
- **Misclassification** — agents unfamiliar with edge cases route tickets to the wrong team, causing additional hand-offs and customer frustration.
- **Inconsistent prioritization** — urgency is subjective; one agent's "high" is another's "medium."

**SmartTicket** addresses this by building an end-to-end machine-learning pipeline that ingests raw ticket data from a relational database, cleans and engineers features from three distinct data modalities (text, numeric, categorical), and trains a classifier to predict both department and priority. The pipeline also applies two feature-selection strategies — a deterministic greedy-forward heuristic and a stochastic genetic algorithm — to identify the most predictive feature subset, reducing dimensionality while preserving (or improving) accuracy.

---

### 2. High-Level System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Raw Support Tickets                        │
│         (ticket text + customer data + metadata)             │
│                  SQLite Database (3 tables)                   │
└───────────────────────────┬──────────────────────────────────┘
                            │
             ┌──────────────┼──────────────────┐
             ▼              ▼                  ▼
     ┌──────────────┐ ┌───────────────┐ ┌──────────────┐
     │     Text     │ │    Numeric    │ │  Categorical │
     │Preprocessing │ │ Preprocessing │ │   Encoding   │
     │              │ │               │ │              │
     │ - Lowercase  │ │ - Median fill │ │ - Normalize  │
     │ - Remove     │ │ - Clip out-   │ │   casing     │
     │   URLs/HTML  │ │   liers       │ │ - One-hot    │
     │ - Stopwords  │ │ - Standard    │ │   encode     │
     │ - Lemmatize  │ │   scaling     │ │              │
     │ - TF-IDF     │ │               │ │              │
     └──────┬───────┘ └───────┬───────┘ └──────┬───────┘
            └─────────────────┼────────────────┘
                              ▼
              ┌───────────────────────────────┐
              │      Feature Engineering      │
              │                               │
              │  word_count, char_count,       │
              │  avg_word_len, sentence_count, │
              │  spend_per_order, return_rate, │
              │  ticket_rate, order_recency    │
              └───────────────┬───────────────┘
                              ▼
              ┌───────────────────────────────┐
              │       Feature Selection       │
              │                               │
              │  Heuristic (Greedy Forward)   │
              │  Genetic Algorithm (GA)       │
              └───────────────┬───────────────┘
                              ▼
              ┌───────────────────────────────┐
              │      KNN Classification       │
              │  department (6 classes)        │
              │  priority   (4 levels)         │
              └───────────────────────────────┘
```

---

### 3. Data Sources and Expected Inputs/Outputs

The dataset is synthetically generated but designed to require real-world preprocessing. It lives in a SQLite database (`smartticket.db`) with three normalized tables:

| Table | Rows | Description |
|---|---|---|
| `tickets` | ~2 500 | Core ticket data — free-text description, channel, product category, region, plus the two target labels (department, priority). Text is intentionally noisy: mixed casing, HTML fragments, URLs, missing values. |
| `customer_metrics` | ~2 500 | Customer history — account age, order count, spend, returns, loyalty tier, satisfaction score. Contains outliers (e.g., `total_spent = 999 999`) and ~3 % NaN values. |
| `ticket_metadata` | ~2 500 | Operational metadata — response time, attachments, replies, escalation flags, sentiment score. Includes invalid values (e.g., negative response times). |

**Messiness by design:** ~5 % of ticket texts are NULL, categorical columns have inconsistent casing (`"Email"` vs `"email"` vs `"EMAIL"`), numeric columns contain outliers outside valid ranges, and ~1 % of ticket IDs are duplicated.

**Input:** Raw ticket text + customer metrics + ticket metadata (24 columns).

**Output:** Predicted `department` (billing, technical, shipping, account, returns, general) and `priority` (low, medium, high, urgent).

---

### 4. AI Components — Implementation Status

| Component | Status | Description |
|---|---|---|
| **Heuristic Feature Selection** | ✅ Complete (Week 4) | Greedy forward selection — iteratively adds the single feature that yields the largest KNN accuracy gain. Deterministic and fast, but may miss beneficial feature interactions. |
| **Genetic Algorithm Feature Selection** | ✅ Complete (Week 4) | Evolutionary search over binary feature masks. Uses tournament-style parent selection, single-point crossover, and bit-flip mutation to explore the combinatorial feature space. Stochastic but capable of discovering non-obvious feature subsets. |
| **Voting Ensemble** | ✅ Complete (Week 6) | Custom `VotingEnsemble` class combining KNN, Decision Tree, Random Forest, Logistic Regression, and SVM via hard voting (majority), soft voting (averaged probabilities), and weighted soft voting (accuracy-proportional weights). |
| **Stacking Ensemble** | ✅ Complete (Week 6) | `StackingClassifier` with 4 base estimators (KNN, Decision Tree, Random Forest, SVM) and Logistic Regression as the meta-learner, trained on 5-fold cross-validated out-of-fold predictions. Learns optimal combination weights from data. |
| **Post-Hoc Explainability (LIME + SHAP)** | ✅ Complete (Week 8) | `MLExplainer` wraps the trained voting ensemble with two model-agnostic explainers: **LIME** (local linear surrogate fit on text perturbations) and **SHAP** (Shapley values over TF-IDF features via `KernelExplainer`). Each prediction is accompanied by a signed word-level attribution table, plus a global aggregate over a batch of tickets. |
| **RL Feedback Loop** | Planned | Reinforcement-learning layer that learns from support-agent corrections to continuously improve classification accuracy over time. |

---

### 5. Architecture — Updated (Week 8)

```
┌─────────────────────────────────────────────────────────────────┐
│                      Raw Support Tickets                         │
│           (ticket text + customer data + metadata)              │
│                    SQLite Database (3 tables)                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
            ┌──────────────┼──────────────────┐
            ▼              ▼                  ▼
    ┌──────────────┐ ┌───────────────┐ ┌──────────────┐
    │     Text     │ │    Numeric    │ │  Categorical │
    │Preprocessing │ │ Preprocessing │ │   Encoding   │
    └──────┬───────┘ └───────┬───────┘ └──────┬───────┘
           └─────────────────┼────────────────┘
                             ▼
             ┌───────────────────────────────┐
             │       Feature Selection       │
             │  Heuristic (Greedy Forward)   │
             │  Genetic Algorithm (GA)       │
             └───────────────┬───────────────┘
                             ▼
             ┌───────────────────────────────┐
             │       Ensemble Modeling       │
             │  VotingEnsemble (hard/soft/   │
             │     weighted)                │
             │  StackingClassifier (LR meta) │
             └───────────────┬───────────────┘
                             ▼
             ┌───────────────────────────────┐
             │  Post-Hoc Explainability      │
             │  MLExplainer                  │
             │  • LIME (local linear fit)    │
             │  • SHAP (Shapley values)      │
             │  • Word-level attributions    │
             │  • Global feature importance  │
             └───────────────┬───────────────┘
                             ▼
             ┌───────────────────────────────┐
             │      Final Classification     │
             │  department (6 classes)       │
             │  priority   (4 levels)        │
             │  + LIME/SHAP attributions     │
             └───────────────────────────────┘
```

---

### 6. Success Criteria

- Pipeline runs end-to-end from raw database to classification results without manual intervention. ✅
- Both feature-selection methods produce competitive accuracy with significantly fewer features than the full baseline. ✅
- All preprocessing steps are logged with before/after statistics for auditability. ✅
- Voting and stacking ensembles outperform the single-model baseline. ✅
- Every classification decision is accompanied by word-level attributions from both LIME and SHAP, showing which tokens in the ticket pushed the model toward (or against) the predicted class. ✅
- LIME and SHAP cross-check each other: when both methods agree on the top-contributing words, the explanation is treated as robust; otherwise it is flagged for closer review. ✅
