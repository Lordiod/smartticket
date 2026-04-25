# SmartTicket — AI-Powered Support Ticket Classification

An end-to-end machine learning pipeline that classifies customer support tickets by **department** (6 classes) and **priority** (4 levels) using text, numeric, and categorical features extracted from a synthetic SQLite database.

Built as a combined deliverable for **Week 3 (Project Proposal)** and **Week 4 (Data Engineering & Feature Selection)**.

---

## What This Project Does

Customer support teams receive hundreds of tickets daily. Manually routing them to the right department and assigning priority is slow and inconsistent. SmartTicket automates this by:

1. **Ingesting** raw ticket data from a SQLite database (3 normalized tables)
2. **Preprocessing** noisy text (URLs, HTML, mixed casing, missing values), outlier-heavy numerics, and inconsistent categoricals
3. **Engineering** derived features (spend per order, return rate, ticket rate, text statistics)
4. **Selecting** the most predictive features using two AI-driven methods:
   - Heuristic greedy forward selection
   - Genetic algorithm (evolutionary search)
5. **Classifying** tickets with KNN and comparing accuracy across methods

---

## Project Structure

```
smartticket/
├── README.md
├── requirements.txt
├── database/
│   ├── generate_database.py       # Creates and seeds the SQLite database
│   └── smartticket.db             # Generated database (~2500 tickets)
├── proposal/
│   └── project_proposal.md        # Week 3 — project proposal
├── pipeline/
│   ├── smartticket_pipeline.py    # Week 4 — full pipeline (script)
│   └── smartticket_pipeline.ipynb # Week 4 — full pipeline (notebook)
└── outputs/
    ├── class_distribution.png     # Department & priority bar charts
    └── comparison_chart.png       # Accuracy comparison across methods
```

---

## The Data

The database (`smartticket.db`) contains **~2500 synthetic support tickets** spread across 3 tables:

### `tickets` — Core ticket data
| Column | Description |
|---|---|
| `ticket_id` | Unique identifier (e.g., TKT-000001) |
| `ticket_text` | Free-text customer message (~5% NULL) |
| `channel` | email, chat, phone, social_media, web_form |
| `product_category` | electronics, clothing, home, food, software, subscription |
| `region` | north_america, europe, asia, middle_east, africa, south_america |
| `department` | **TARGET** — billing, technical, shipping, account, returns, general |
| `priority` | **TARGET** — low, medium, high, urgent |

### `customer_metrics` — Customer history
| Column | Description |
|---|---|
| `account_age_days` | Days since account creation |
| `total_orders` | Lifetime order count |
| `total_spent` | Lifetime spend ($) |
| `returns_count` | Number of returns |
| `avg_order_value` | Average order value ($) |
| `days_since_last_order` | Recency of last order |
| `loyalty_tier` | bronze, silver, gold, platinum |
| `previous_tickets` | Number of past support tickets |
| `avg_response_satisfaction` | Satisfaction score (1.0–5.0) |

### `ticket_metadata` — Operational metadata
| Column | Description |
|---|---|
| `response_time_hours` | Hours until first response |
| `num_attachments` | Files attached to ticket |
| `num_replies` | Number of replies in thread |
| `escalated` | Whether ticket was escalated (0/1) |
| `reopened` | Whether ticket was reopened (0/1) |
| `sentiment_score` | Pre-computed sentiment (-1.0 to 1.0) |
| `word_count_raw` | Raw word count of ticket text |
| `has_order_number` | Whether an order number was referenced (0/1) |

### Built-In Data Messiness

The data was intentionally generated with real-world noise to make preprocessing necessary:

- **Missing values**: ~5% of ticket texts are NULL, ~3% of numeric columns have NaN
- **Text noise**: ALL CAPS, extra punctuation (!!!, ???), URLs, email addresses, HTML fragments (`&amp;`, `<br>`), emoji-like text (`:)`, `<3`), double spaces, newlines
- **Numeric outliers**: `total_spent` at 999,999; negative `response_time_hours`; `satisfaction` scores of 10.0 (max should be 5.0); `account_age_days` at 99,999
- **Categorical inconsistency**: Mixed casing — "Email", "email", "EMAIL" all appear
- **Duplicate records**: ~1% duplicate `ticket_id` values
- **Realistic correlations**: Billing tickets have higher spend; technical tickets skew toward electronics/software; returns tickets have higher return counts; urgent tickets contain words like "ASAP", "immediately", "hacked"

---

## Pipeline Steps

| Step | What It Does |
|---|---|
| **Data Ingestion** | JOIN 3 tables from SQLite, remove duplicate ticket_ids |
| **Text Preprocessing** | Fill NULLs, lowercase, strip URLs/HTML/emails, remove stopwords, lemmatize |
| **Text Feature Engineering** | Extract word_count, char_count, avg_word_len, sentence_count |
| **Numeric Preprocessing** | Median-fill NaNs, clip outliers to valid ranges |
| **Derived Features** | Compute spend_per_order, return_rate, ticket_rate, order_recency_score |
| **Categorical Encoding** | Standardize casing, one-hot encode channel/product/region/loyalty |
| **Feature Matrix** | Combine TF-IDF (500 features) + scaled numerics + one-hot encoded categoricals |
| **Baseline KNN** | Train KNN on all 357 features |
| **Heuristic Selection** | Greedy forward — add one feature at a time by accuracy gain (max 10) |
| **Genetic Algorithm** | Evolutionary search — population of 20, 15 generations, 5% mutation |

---

## Results

```
Method                    Accuracy        # Features
------------------------------------------------------------
Baseline (all+TF-IDF)     0.6452          357
Heuristic (Greedy)        0.8116          8
Genetic Algorithm         0.7026          21
```

Both feature selection methods outperformed the baseline while using far fewer features. The heuristic approach achieved the best result — **81.16% accuracy with just 8 features** — demonstrating that a small set of well-chosen features beats a massive noisy feature space for KNN classification.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate the database

```bash
python database/generate_database.py
```

### 3. Run the pipeline

**Script:**
```bash
python pipeline/smartticket_pipeline.py
```

**Notebook:**
```bash
jupyter notebook pipeline/smartticket_pipeline.ipynb
```

Both produce identical output. Charts are saved to `outputs/`.

---

## Tech Stack

- **Python 3.11+**
- **pandas / numpy** — data manipulation
- **scikit-learn** — KNN classifier, TF-IDF, StandardScaler, LabelEncoder, train/test split
- **NLTK** — stopword removal, lemmatization
- **scipy** — sparse matrix operations
- **matplotlib / seaborn** — visualization
- **SQLite** — data storage
