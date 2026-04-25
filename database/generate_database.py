"""
SmartTicket — Synthetic Database Generator
Creates a SQLite database with ~2500 noisy support tickets across 3 tables.
"""

import sqlite3
import random
import string
import os
import numpy as np
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_TICKETS = 2500
DUPLICATE_RATE = 0.01
MISSING_TEXT_RATE = 0.05
MISSING_NUMERIC_RATE = 0.03
OUTLIER_RATE = 0.03

DEPARTMENTS = ["billing", "technical", "shipping", "account", "returns", "general"]
PRIORITIES = ["low", "medium", "high", "urgent"]
CHANNELS = ["email", "chat", "phone", "social_media", "web_form"]
PRODUCTS = ["electronics", "clothing", "home", "food", "software", "subscription"]
REGIONS = ["north_america", "europe", "asia", "middle_east", "africa", "south_america"]
LOYALTY_TIERS = ["bronze", "silver", "gold", "platinum"]

# ---------------------------------------------------------------------------
# Helper generators
# ---------------------------------------------------------------------------

def rand_date(start_year=2023, end_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    return (start + timedelta(days=random.randint(0, (end - start).days))).strftime("%Y-%m-%d")

def rand_order():
    return str(random.randint(100000, 999999))

def rand_amount():
    return f"{random.uniform(9.99, 499.99):.2f}"

def rand_card():
    return "".join(random.choices(string.digits, k=4))

def rand_email():
    names = ["john", "jane", "mike", "sara", "alex", "pat", "kim", "lee", "sam", "taylor"]
    domains = ["gmail.com", "yahoo.com", "outlook.com", "company.co"]
    return f"{random.choice(names)}{random.randint(1,999)}@{random.choice(domains)}"

def rand_device():
    return random.choice(["iPhone 14", "Samsung Galaxy S23", "iPad Pro", "Pixel 7", "MacBook Air", "Dell XPS 15"])

def rand_os():
    return random.choice(["iOS 17", "Android 14", "Windows 11", "macOS Sonoma", "Ubuntu 22.04"])

def rand_error_code():
    return f"ERR-{random.randint(1000,9999)}"

def rand_product():
    return random.choice(["Bluetooth Speaker", "Wireless Headphones", "Smart Watch", "Laptop Stand",
                           "USB-C Hub", "Mechanical Keyboard", "Webcam HD", "Fitness Tracker",
                           "Portable Charger", "Noise Cancelling Earbuds", "Cotton T-Shirt",
                           "Running Shoes", "Winter Jacket", "Yoga Mat", "Coffee Maker"])

def rand_location():
    return random.choice(["Memphis TN", "Louisville KY", "Chicago IL", "Los Angeles CA",
                           "Frankfurt Germany", "Shenzhen China", "Toronto Canada"])

def rand_tracking():
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=12))

def rand_company():
    return random.choice(["Acme Corp", "TechStart Inc", "GlobalTrade LLC", "DataViz Co", "BrightPath Ltd"])

# ---------------------------------------------------------------------------
# Ticket text templates per department
# ---------------------------------------------------------------------------

TEMPLATES = {
    "billing": [
        lambda: f"I was charged twice for order #{rand_order()}. The duplicate charge of ${rand_amount()} appeared on {rand_date()}. Please refund ASAP.",
        lambda: f"My credit card ending in {rand_card()} was billed ${rand_amount()} but I never placed this order. This is unauthorized!!",
        lambda: f"Hi, I need an invoice for order #{rand_order()} for my company {rand_company()}. Tax ID: {random.randint(10,99)}-{random.randint(1000000,9999999)}. Thanks",
        lambda: f"The promo code SAVE{random.randint(10,50)} was supposed to give me {random.randint(10,40)}% off but I was charged full price ${rand_amount()}",
        lambda: f"Why am I being charged ${rand_amount()}/month?? I cancelled my subscription on {rand_date()}!!!",
        lambda: f"I see a pending charge of ${rand_amount()} on my statement from {rand_date()}. Order #{rand_order()}. Can you explain what this is for?",
    ],
    "technical": [
        lambda: f"The app keeps crashing on my {rand_device()} running {rand_os()}. Error: {rand_error_code()}. Tried reinstalling 3 times already.",
        lambda: f"Can't connect my {rand_product()} to wifi. Firmware version {random.randint(1,5)}.{random.randint(0,9)}.{random.randint(0,9)}. LED is blinking {random.choice(['red', 'orange', 'blue'])}.",
        lambda: f"Getting error '{rand_error_code()} — connection timeout' when trying to sync data. Screenshot attached. Been broken since {rand_date()}.",
        lambda: f"My {rand_product()} stopped working after the latest update. Model: {random.choice(['X100','Z200','Pro-5','Elite-3'])}, Serial: SN-{random.randint(100000,999999)}",
        lambda: f"Software license key LK-{''.join(random.choices(string.ascii_uppercase+string.digits, k=16))} isn't activating. Says 'already in use' but I only have one device.",
        lambda: f"Blue screen of death every time I open the application on {rand_os()}. Error code: {rand_error_code()}. Really need this fixed for work.",
    ],
    "shipping": [
        lambda: f"Order #{rand_order()} was supposed to arrive by {rand_date()} but tracking shows it's still in {rand_location()}. Very frustrated.",
        lambda: f"Received wrong item. Ordered {rand_product()} but got {rand_product()}. Order #{rand_order()}. Need correct item shipped ASAP",
        lambda: f"Package arrived completely damaged. {rand_product()} is broken. Photos attached. Order #{rand_order()} needs replacement.",
        lambda: f"Tracking number {rand_tracking()} hasn't updated in {random.randint(3,14)} days. Is my package lost??",
        lambda: f"I need to change the delivery address for order #{rand_order()} from {rand_location()} to {rand_location()}. Urgent!",
        lambda: f"My order #{rand_order()} shows delivered but I never received it. Checked with neighbors too. Need help locating package.",
    ],
    "account": [
        lambda: f"Can't log in to my account {rand_email()}. Password reset email never arrives. Checked spam folder.",
        lambda: f"Someone changed my account email to {rand_email()}. I think I was hacked. Please help immediately!",
        lambda: f"I want to delete my account and all my data. Email: {rand_email()}. GDPR request.",
        lambda: f"Need to update my shipping address and phone number on file. New address: {random.randint(100,9999)} {random.choice(['Oak','Elm','Main','Park'])} St, {rand_location()}",
        lambda: f"How do I merge my two accounts? I have {rand_email()} and {rand_email()} with separate order histories.",
        lambda: f"My account shows the wrong name. It says {random.choice(['John','Jane','Alex'])} but my name is {random.choice(['Michael','Sarah','Chris'])}. Please fix this.",
    ],
    "returns": [
        lambda: f"I want to return {rand_product()} from order #{rand_order()}. Bought it {random.randint(3,28)} days ago. Reason: {random.choice(['defective', 'wrong size', 'not as described', 'changed my mind'])}",
        lambda: f"Return label for order #{rand_order()} doesn't work. Getting error when scanning. Need new label.",
        lambda: f"I returned {rand_product()} via {random.choice(['UPS','FedEx','USPS','DHL'])} on {rand_date()} (tracking: {rand_tracking()}) but haven't received refund yet.",
        lambda: f"Is this item eligible for return? Product: {rand_product()}, ordered {random.randint(5,45)} days ago, condition: {random.choice(['unopened', 'like new', 'used once', 'slightly worn'])}",
        lambda: f"Exchange request: want to swap {rand_product()} (size {random.choice(['S','M','L','XL'])}) for {rand_product()} (size {random.choice(['S','M','L','XL'])}) from order #{rand_order()}",
    ],
    "general": [
        lambda: f"Do you ship to {random.choice(['Australia', 'Brazil', 'Japan', 'India', 'Mexico', 'Nigeria'])}? Interested in {rand_product()} but can't find shipping info on your website.",
        lambda: f"What's your return policy for {random.choice(PRODUCTS)} items? And do you offer gift wrapping?",
        lambda: f"When will {rand_product()} be back in stock? I've been waiting for {random.randint(2,8)} weeks now.",
        lambda: f"Just wanted to say your customer service was amazing last time. {random.choice(['Sarah','Mike','Alex','Jordan','Pat'])} really helped me out!",
        lambda: f"Hi, do you have any current promotions or discount codes? Looking at your {random.choice(PRODUCTS)} products.",
        lambda: f"I'm a first time buyer, wondering about your warranty policy for {rand_product()}. How long does it cover?",
    ],
}

# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------

NOISE_INJECTORS = [
    lambda t: t.upper(),                                          # ALL CAPS
    lambda t: t + "!!!",                                          # extra punctuation
    lambda t: t + " " + random.choice([":(", ":)", "<3", ":/", ";)"]),  # emoji-like
    lambda t: t + f" see http://myorder.com/status/{random.randint(10000,99999)}",  # URL
    lambda t: t + f"  Contact me at {rand_email()}  ",            # email + whitespace
    lambda t: t.replace(".", "...", 1),                           # ellipsis
    lambda t: "  " + t + "  ",                                   # leading/trailing whitespace
    lambda t: t.replace(". ", ".\n", 1),                          # newline injection
    lambda t: t + " &amp; " + "thank you &nbsp;",                 # HTML entities
    lambda t: t + " <br> please respond <b>quickly</b>",          # HTML tags
    lambda t: t.replace(" ", "  ", 2),                            # double spaces
]

def inject_noise(text):
    """Apply 1-3 random noise transformations."""
    n = random.randint(1, 3)
    for fn in random.sample(NOISE_INJECTORS, min(n, len(NOISE_INJECTORS))):
        text = fn(text)
    return text


def casing_variant(value):
    """Randomly vary casing for categorical inconsistency."""
    r = random.random()
    if r < 0.1:
        return value.upper()
    elif r < 0.2:
        return value.capitalize()
    return value


# ---------------------------------------------------------------------------
# Record generation
# ---------------------------------------------------------------------------

def generate_ticket(idx):
    """Generate one ticket + metrics + metadata row."""
    dept = random.choice(DEPARTMENTS)
    # Priority correlations
    if dept == "account" and random.random() < 0.3:
        priority = "urgent"
    elif dept == "general":
        priority = random.choices(PRIORITIES, weights=[0.50, 0.35, 0.10, 0.05])[0]
    else:
        priority = random.choices(PRIORITIES, weights=[0.20, 0.40, 0.25, 0.15])[0]

    ticket_id = f"TKT-{idx:06d}"

    # Text
    template_fn = random.choice(TEMPLATES[dept])
    ticket_text = template_fn()

    # Inject noise into ~40% of texts
    if random.random() < 0.40:
        ticket_text = inject_noise(ticket_text)

    # Urgency keywords for urgent tickets
    if priority == "urgent" and random.random() < 0.6:
        ticket_text += " " + random.choice(["ASAP", "immediately", "THIS IS URGENT", "hacked", "unauthorized"])

    # Missing text
    if random.random() < MISSING_TEXT_RATE:
        ticket_text = None

    # Channel / product / region with casing inconsistency
    channel = casing_variant(random.choice(CHANNELS)) if random.random() > 0.03 else None
    region = casing_variant(random.choice(REGIONS))

    # Product correlations
    if dept == "technical":
        product_category = casing_variant(random.choices(
            PRODUCTS, weights=[0.35, 0.02, 0.03, 0.01, 0.40, 0.19])[0])
    else:
        product_category = casing_variant(random.choice(PRODUCTS)) if random.random() > 0.03 else None

    created_at = rand_date()

    # --- Customer metrics ---
    account_age_days = int(np.random.lognormal(6, 1))
    if random.random() < OUTLIER_RATE:
        account_age_days = 99999

    total_orders = max(0, int(np.random.exponential(10)))
    total_spent = round(total_orders * np.random.uniform(15, 80), 2)
    if dept == "billing":
        total_spent *= random.uniform(1.5, 3.0)
        total_spent = round(total_spent, 2)
    if random.random() < OUTLIER_RATE:
        total_spent = 999999.0

    returns_count = max(0, int(np.random.exponential(2)))
    if dept == "returns":
        returns_count += random.randint(2, 8)

    avg_order_value = round(total_spent / (total_orders + 1), 2)

    days_since_last_order = max(0, int(np.random.exponential(30)))
    if dept == "shipping":
        days_since_last_order = random.randint(0, 7)

    loyalty_tier = random.choice(LOYALTY_TIERS)

    previous_tickets = max(0, int(np.random.exponential(3)))
    if priority == "urgent":
        previous_tickets += random.randint(2, 6)

    avg_response_satisfaction = round(np.random.uniform(1.0, 5.0), 1)
    if dept == "technical":
        avg_response_satisfaction = round(max(1.0, avg_response_satisfaction - random.uniform(0.5, 1.5)), 1)
    if random.random() < OUTLIER_RATE:
        avg_response_satisfaction = 10.0  # invalid

    # --- Metadata ---
    response_time_hours = round(np.random.exponential(12), 1)
    if random.random() < OUTLIER_RATE:
        response_time_hours = -round(random.uniform(1, 10), 1)  # negative

    num_attachments = random.choices([0, 1, 2, 3, 4], weights=[0.5, 0.25, 0.15, 0.07, 0.03])[0]
    num_replies = max(0, int(np.random.exponential(3)))
    escalated = 1 if (priority in ["high", "urgent"] and random.random() < 0.4) else (1 if random.random() < 0.05 else 0)
    reopened = 1 if random.random() < 0.08 else 0

    sentiment_score = round(np.random.uniform(-1.0, 1.0), 2)
    if priority == "low" or dept == "general":
        sentiment_score = round(abs(sentiment_score), 2)

    word_count_raw = len(ticket_text.split()) if ticket_text else 0
    has_order_number = 1 if (dept in ["billing", "shipping", "returns"] and random.random() < 0.85) else (1 if random.random() < 0.15 else 0)

    # Inject NaN into some numeric fields
    metrics = {
        "account_age_days": account_age_days,
        "total_orders": total_orders,
        "total_spent": total_spent,
        "returns_count": returns_count,
        "avg_order_value": avg_order_value,
        "days_since_last_order": days_since_last_order,
        "previous_tickets": previous_tickets,
        "avg_response_satisfaction": avg_response_satisfaction,
    }
    metadata = {
        "response_time_hours": response_time_hours,
        "num_attachments": num_attachments,
        "num_replies": num_replies,
        "escalated": escalated,
        "reopened": reopened,
        "sentiment_score": sentiment_score,
        "word_count_raw": word_count_raw,
        "has_order_number": has_order_number,
    }

    for d in [metrics, metadata]:
        for k in list(d.keys()):
            if random.random() < MISSING_NUMERIC_RATE:
                d[k] = None

    return {
        "ticket": (ticket_id, created_at, f"CUST-{random.randint(1000,9999)}", ticket_text,
                   channel, product_category, region, dept, priority),
        "metrics": (ticket_id, metrics["account_age_days"], metrics["total_orders"],
                    metrics["total_spent"], metrics["returns_count"], metrics["avg_order_value"],
                    metrics["days_since_last_order"], loyalty_tier,
                    metrics["previous_tickets"], metrics["avg_response_satisfaction"]),
        "metadata": (ticket_id, metadata["response_time_hours"], metadata["num_attachments"],
                     metadata["num_replies"], metadata["escalated"], metadata["reopened"],
                     metadata["sentiment_score"], metadata["word_count_raw"],
                     metadata["has_order_number"]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smartticket.db")

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create tables
    cur.executescript("""
        CREATE TABLE tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            customer_id TEXT NOT NULL,
            ticket_text TEXT,
            channel TEXT,
            product_category TEXT,
            region TEXT,
            department TEXT NOT NULL,
            priority TEXT NOT NULL
        );

        CREATE TABLE customer_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id TEXT NOT NULL REFERENCES tickets(ticket_id),
            account_age_days INTEGER,
            total_orders INTEGER,
            total_spent REAL,
            returns_count INTEGER,
            avg_order_value REAL,
            days_since_last_order INTEGER,
            loyalty_tier TEXT,
            previous_tickets INTEGER,
            avg_response_satisfaction REAL
        );

        CREATE TABLE ticket_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id TEXT NOT NULL REFERENCES tickets(ticket_id),
            response_time_hours REAL,
            num_attachments INTEGER,
            num_replies INTEGER,
            escalated INTEGER,
            reopened INTEGER,
            sentiment_score REAL,
            word_count_raw INTEGER,
            has_order_number INTEGER
        );
    """)

    # Generate records
    records = []
    for i in range(1, NUM_TICKETS + 1):
        records.append(generate_ticket(i))

    # Inject ~1% duplicates (reuse existing ticket_ids)
    num_dupes = int(NUM_TICKETS * DUPLICATE_RATE)
    for _ in range(num_dupes):
        source = random.choice(records)
        dupe = generate_ticket(random.randint(1, NUM_TICKETS))  # new data, old id
        # Overwrite ticket_id with the source's
        old_tid = source["ticket"]["0"] if isinstance(source["ticket"], dict) else source["ticket"][0]
        t = list(dupe["ticket"]); t[0] = old_tid; dupe["ticket"] = tuple(t)
        m = list(dupe["metrics"]); m[0] = old_tid; dupe["metrics"] = tuple(m)
        md = list(dupe["metadata"]); md[0] = old_tid; dupe["metadata"] = tuple(md)
        records.append(dupe)

    # Insert
    for rec in records:
        cur.execute(
            "INSERT INTO tickets (ticket_id, created_at, customer_id, ticket_text, channel, product_category, region, department, priority) VALUES (?,?,?,?,?,?,?,?,?)",
            rec["ticket"],
        )
        cur.execute(
            "INSERT INTO customer_metrics (ticket_id, account_age_days, total_orders, total_spent, returns_count, avg_order_value, days_since_last_order, loyalty_tier, previous_tickets, avg_response_satisfaction) VALUES (?,?,?,?,?,?,?,?,?,?)",
            rec["metrics"],
        )
        cur.execute(
            "INSERT INTO ticket_metadata (ticket_id, response_time_hours, num_attachments, num_replies, escalated, reopened, sentiment_score, word_count_raw, has_order_number) VALUES (?,?,?,?,?,?,?,?,?)",
            rec["metadata"],
        )

    conn.commit()

    # --- Summary ---
    print("=" * 60)
    print("  SmartTicket Database Generated Successfully")
    print("=" * 60)
    print(f"\nDatabase: {db_path}\n")

    for table in ["tickets", "customer_metrics", "ticket_metadata"]:
        count = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows")

    print("\nDepartment distribution:")
    for row in cur.execute("SELECT department, COUNT(*) FROM tickets GROUP BY department ORDER BY COUNT(*) DESC"):
        print(f"  {row[0]:<15} {row[1]}")

    print("\nPriority distribution:")
    for row in cur.execute("SELECT priority, COUNT(*) FROM tickets GROUP BY priority ORDER BY COUNT(*) DESC"):
        print(f"  {row[0]:<15} {row[1]}")

    # Null text count
    null_text = cur.execute("SELECT COUNT(*) FROM tickets WHERE ticket_text IS NULL").fetchone()[0]
    print(f"\nNull ticket_text: {null_text}")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
