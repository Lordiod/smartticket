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

### 4. Planned AI Components

| Component | Status | Description |
|---|---|---|
| **Heuristic Feature Selection** | Week 4 | Greedy forward selection — iteratively adds the single feature that yields the largest KNN accuracy gain. Deterministic and fast, but may miss beneficial feature interactions. |
| **Genetic Algorithm Feature Selection** | Week 4 | Evolutionary search over binary feature masks. Uses tournament-style parent selection, single-point crossover, and bit-flip mutation to explore the combinatorial feature space. Stochastic but capable of discovering non-obvious feature subsets. |
| **Ensemble Modeling** | Future | Voting and stacking classifiers combining KNN, Random Forest, and SVM to improve robustness. |
| **Logic-Based Explainability** | Future | Decision-rule extraction to provide human-readable explanations for each classification decision. |
| **RL Feedback Loop** | Future | Reinforcement-learning layer that learns from support-agent corrections to continuously improve classification accuracy over time. |

---

### 5. Success Criteria

- Pipeline runs end-to-end from raw database to classification results without manual intervention.
- Both feature-selection methods produce competitive accuracy with significantly fewer features than the full baseline.
- All preprocessing steps are logged with before/after statistics for auditability.
