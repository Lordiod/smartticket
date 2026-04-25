"""
SmartTicket — Explainability & Reasoning Layer
Week 8 Deliverable

Logic-based assistant that integrates with ML predictions to provide:
  • Rule-based verification and optional override of ML classifications
  • Human-readable explanations for every decision
  • Full reasoning trace showing which rules fired and why
  • Confidence scoring that reflects neuro-symbolic agreement

Architecture (Neuro-Symbolic Pipeline):
    ┌────────────┐     ┌──────────────────┐     ┌──────────────────────┐
    │  Raw Text  │────►│  ML Classifiers  │────►│                      │
    │ + Metadata │     │  (KNN / RF / SVM │     │   Logic Layer        │
    └────────────┘     │   / Ensemble)    │     │  TicketExplainer     │──► ExplanationResult
         │             └──────────────────┘     │                      │
         └──────────────────────────────────────►│  (rules + ML pred)   │
                                                └──────────────────────┘

Short Report — How Logic Improves Interpretability and Control
--------------------------------------------------------------
1. TRANSPARENCY  — Every decision comes with a plain-English trace of which
   keyword/threshold rules fired, what evidence triggered them, and the
   resulting confidence.  Stakeholders can audit classifications without
   understanding neural network internals.

2. CONTROL  — Domain experts add, tune, or delete rules in minutes, without
   re-training any model.  A known failure mode (e.g., "hacked" tickets
   misclassified as 'billing') is fixed instantly by editing a rule.

3. OVERRIDE & SAFETY-NETTING  — High-confidence rules (≥ 90% by default)
   override the ML prediction when they detect clear domain signals (e.g.,
   "unauthorized access" → account/urgent).  Priority is only ever escalated
   by logic, never downgraded, creating a safety floor.

4. HUMAN-IN-THE-LOOP FLAGGING  — When ML and logic strongly disagree (both
   highly confident but reaching different conclusions), the ticket is flagged
   for human review rather than silently misclassified.

5. NEURO-SYMBOLIC SYNERGY  — When ML and logic agree, the confidence score
   is boosted, enabling downstream automation with higher certainty.  High-
   agreement tickets can be auto-routed; disagreement routes to a review queue.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple


# ══════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════

@dataclass
class RuleFiring:
    """Records a single symbolic rule that fired during evaluation."""
    rule_name: str
    rule_type: str            # "department" | "priority" | "override"
    conclusion: str           # predicted label
    confidence: float
    matched_evidence: List[str]
    explanation: str


@dataclass
class ExplanationResult:
    """Full explanation for a single ticket classification decision."""

    # Original ML predictions (may be None if called standalone)
    ml_department: Optional[str]
    ml_priority: Optional[str]

    # Final decisions — may differ from ML if a rule fires an override
    final_department: str
    final_priority: str

    # Source of each final decision
    dept_source: str      # "ml" | "ml+logic" | "logic_override" | "logic_only" | "default"
    priority_source: str

    # Confidence (0–1)
    dept_confidence: float
    priority_confidence: float

    # Reasoning
    rules_fired: List[RuleFiring] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)

    # Agreement flags
    dept_agreement: bool = True
    priority_agreement: bool = True

    # Review flags
    needs_human_review: bool = False
    review_reason: str = ""

    def summary(self) -> str:
        """Return a concise human-readable summary of this classification."""
        lines = [
            "┌─ SmartTicket Classification ─────────────────────────────┐",
            f"│  Department : {self.final_department.upper():<15}"
            f"  ({self.dept_source}, {self.dept_confidence:.0%})         │",
            f"│  Priority   : {self.final_priority.upper():<15}"
            f"  ({self.priority_source}, {self.priority_confidence:.0%})         │",
        ]
        if self.rules_fired:
            lines.append(f"│  Rules fired: {len(self.rules_fired):<47}│")
            for r in self.rules_fired[:5]:
                ev = ", ".join(r.matched_evidence[:2]) if r.matched_evidence else "—"
                line = f"│    [{r.rule_type[:4]}] {r.rule_name}: {ev}"
                lines.append(f"{line:<62}│")
        if self.needs_human_review:
            lines.append(f"│  ⚠ REVIEW : {self.review_reason[:50]:<50}│")
        lines.append("└──────────────────────────────────────────────────────────┘")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# Symbolic Rule Definitions
# ══════════════════════════════════════════════════════════════

# Department keyword rules — each entry:
#   (rule_name, [regex_patterns], department_label, base_confidence, explanation)
DEPT_KEYWORD_RULES: List[Tuple] = [
    (
        "billing_payment",
        [
            r"\bcharg(ed|e|es)\b", r"\brefund\b", r"\binvoice\b", r"\bpayment\b",
            r"\bbilling\b", r"\bcredit.?card\b", r"\bsubscription.?fee\b",
            r"\bduplicate.?charge\b", r"\bbilled\b", r"\bunauthorized.?charge\b",
            r"\bstatement\b", r"\btransaction\b", r"\bpromo.?code\b",
        ],
        "billing", 0.88,
        "Ticket contains billing or payment-related keywords",
    ),
    (
        "technical_error",
        [
            r"\berror.?code\b", r"\bcrash(ing|ed)?\b", r"\bfirmware\b",
            r"\bblue.?screen\b", r"\bfreez(es|ing|ed)\b", r"\bbsod\b",
            r"\bnot.?working\b", r"\binstallation\b", r"\blicense.?key\b",
            r"\bactivat(e|ion|ing)\b", r"\bsoftware.?bug\b", r"\bapp.+crash\b",
            r"\bwifi\b", r"\bconnection.?timeout\b", r"\bsync\b",
        ],
        "technical", 0.86,
        "Ticket describes a technical or software problem",
    ),
    (
        "shipping_delivery",
        [
            r"\btracking\b", r"\bdelivery\b", r"\bshipping\b", r"\bpackage\b",
            r"\btracking.?number\b", r"\bin.?transit\b", r"\barriv(ed|al|e)\b",
            r"\bdamaged.?(item|product|package)\b", r"\bwrong.?item\b",
            r"\blost.?package\b", r"\bdelivery.?address\b", r"\border.+deliver\b",
        ],
        "shipping", 0.87,
        "Ticket relates to shipment, delivery, or package status",
    ),
    (
        "account_access",
        [
            r"\bcan.?t\s+log\s*(in|into)\b", r"\bpassword.?reset\b",
            r"\bhacked\b", r"\bcompromised\b", r"\baccount.?access\b",
            r"\bdelete.?account\b", r"\bgdpr\b", r"\bmerge.?account\b",
            r"\bwrong.?name\b", r"\bunauthorized.?access\b", r"\blogin\b",
            r"\bsign.?in\b",
        ],
        "account", 0.86,
        "Ticket involves account access, security, or identity issues",
    ),
    (
        "returns_exchange",
        [
            r"\breturn(ing|ed|s)?\b", r"\bexchange\b", r"\brefund.?status\b",
            r"\breturn.?label\b", r"\bsend.?back\b", r"\bwant.?to.?return\b",
            r"\bproduct.?(defective|damaged|broken)\b", r"\brepla(ce|cement)\b",
            r"\breceipt\b", r"\bitem.+return\b",
        ],
        "returns", 0.85,
        "Ticket is about returning or exchanging a product",
    ),
    (
        "general_inquiry",
        [
            r"\bdo.?you.?ship\b", r"\breturn.?policy\b", r"\bin.?stock\b",
            r"\bback.?in.?stock\b", r"\bdiscount.?code\b", r"\bpromotion\b",
            r"\bgift.?wrapp\b", r"\bwarranty.?policy\b", r"\bfirst.?time\b",
            r"\bupgrade.?plan\b", r"\bquick.?question\b", r"\bwhen.+available\b",
        ],
        "general", 0.75,
        "Ticket is a general inquiry or policy question",
    ),
]

# Priority keyword rules — (rule_name, [patterns], priority_label, base_confidence, explanation)
PRIORITY_KEYWORD_RULES: List[Tuple] = [
    (
        "urgent_alarm",
        [
            r"\burgent\b", r"\basap\b", r"\bimmediately\b", r"\bemergency\b",
            r"\bhacked\b", r"\bunauthorized\b", r"\bfraud\b", r"\bstolen\b",
            r"\bcritical\b", r"\bthis.?is.?urgent\b", r"\bhelp.?immediately\b",
        ],
        "urgent", 0.93,
        "Ticket contains strong urgency or security-alarm keywords",
    ),
    (
        "high_impact",
        [
            r"\b(completely|totally)\s+(broken|down|failed)\b",
            r"\bdown\b.{0,30}\bproduction\b", r"\bcannot\s+(work|function|access)\b",
            r"\bbusiness.?(impact|critical)\b", r"\bdata.?loss\b",
            r"\bsecurity\b", r"\bblue.?screen\b",
        ],
        "high", 0.80,
        "Ticket indicates a high-impact problem affecting work or security",
    ),
    (
        "medium_frustration",
        [
            r"!!+", r"\bfrustrat(ed|ing)\b", r"\bstill.+not.+fix\b",
            r"\btried.+times\b", r"\bdisappointed\b", r"\bnot.?happy\b",
            r"\bunacceptable\b", r"\bkeeps.+happen\b",
        ],
        "medium", 0.70,
        "Ticket shows user frustration suggesting medium priority",
    ),
    (
        "low_inquiry",
        [
            r"\bjust\s+wondering\b", r"\bquick\s+question\b",
            r"\bdo\s+you\b", r"\bwhen\s+will\b", r"\bwhat.?is\s+your\b",
            r"\bthank\s+you\b", r"\bappreciate\b", r"\bgeneral\s+question\b",
            r"\bjust\s+wanted\s+to\s+say\b",
        ],
        "low", 0.72,
        "Ticket appears to be a low-urgency inquiry or positive feedback",
    ),
]

# Override rules — fire before ML comparison; can force dept and/or priority
# condition(text_lower, metadata_dict) → bool
OVERRIDE_RULES: List[Dict] = [
    {
        "name": "security_breach_override",
        "condition": lambda t, m: bool(
            re.search(r"\b(hacked|unauthorized.?access|account.?comprom|fraudulent|identity.?theft)\b", t)
        ),
        "department": "account",
        "priority": "urgent",
        "confidence": 0.97,
        "explanation": "Security breach detected — forces account department + urgent priority",
    },
    {
        "name": "duplicate_charge_override",
        "condition": lambda t, m: bool(
            re.search(r"\b(double.?charg|charg.{1,10}twice|unauthorized.?charg)\b", t)
        ),
        "department": "billing",
        "priority": "high",
        "confidence": 0.95,
        "explanation": "Unauthorized/duplicate charge detected — billing + high priority",
    },
    {
        "name": "escalated_ticket_override",
        "condition": lambda t, m: m.get("escalated", 0) == 1 if m else False,
        "department": None,  # don't override department
        "priority": "high",
        "confidence": 0.90,
        "explanation": "Ticket is flagged escalated — enforces minimum high priority",
    },
    {
        "name": "lost_package_override",
        "condition": lambda t, m: bool(
            re.search(r"\b(lost.?package|package.?lost|missing.?order|never.?arrived|never.?received)\b", t)
        ),
        "department": "shipping",
        "priority": "high",
        "confidence": 0.92,
        "explanation": "Lost/missing package — forces shipping department + high priority",
    },
    {
        "name": "high_reply_count_override",
        "condition": lambda t, m: m.get("num_replies", 0) >= 5 if m else False,
        "department": None,
        "priority": "high",
        "confidence": 0.88,
        "explanation": "Ticket has 5+ replies — long unresolved conversation, escalate priority",
    },
]

# Priority order for comparison (higher = more severe)
PRIORITY_ORDER = {"low": 0, "medium": 1, "high": 2, "urgent": 3}
DEPARTMENTS = ["billing", "technical", "shipping", "account", "returns", "general"]
PRIORITIES = ["low", "medium", "high", "urgent"]


# ══════════════════════════════════════════════════════════════
# TicketExplainer
# ══════════════════════════════════════════════════════════════

class TicketExplainer:
    """
    Logic-based assistant that explains and optionally corrects ML ticket
    classifications by applying a symbolic rule set.

    Parameters
    ----------
    override_threshold : float (default 0.90)
        Minimum rule confidence required to override an ML prediction.
    review_on_disagreement : bool (default True)
        Flag tickets for human review when ML and logic strongly disagree.
    """

    def __init__(
        self,
        override_threshold: float = 0.90,
        review_on_disagreement: bool = True,
    ):
        self.override_threshold = override_threshold
        self.review_on_disagreement = review_on_disagreement

    # ── Core explanation method ────────────────────────────────

    def explain(
        self,
        text: str,
        ml_department: Optional[str] = None,
        ml_priority: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> ExplanationResult:
        """
        Classify and explain a single ticket.

        Parameters
        ----------
        text : str
            Raw (uncleaned) ticket text.
        ml_department : str or None
            Department label predicted by the ML model.
        ml_priority : str or None
            Priority label predicted by the ML model.
        metadata : dict or None
            Optional metadata dict (keys: escalated, num_replies, sentiment_score, …).

        Returns
        -------
        ExplanationResult
        """
        text_lower = str(text).lower() if text else ""
        metadata = metadata or {}
        rules_fired: List[RuleFiring] = []
        trace: List[str] = []

        # ── 1. Department keyword rules ────────────────────────
        dept_scores: Dict[str, float] = {}
        for rule_name, patterns, dept, base_conf, expl in DEPT_KEYWORD_RULES:
            matched_pats = [p for p in patterns if re.search(p, text_lower)]
            if matched_pats:
                # Confidence scales slightly with number of distinct matches
                conf = min(base_conf + 0.015 * (len(matched_pats) - 1), 0.99)
                dept_scores[dept] = max(dept_scores.get(dept, 0.0), conf)
                evidence = self._clean_patterns(matched_pats[:3])
                rules_fired.append(RuleFiring(
                    rule_name=rule_name,
                    rule_type="department",
                    conclusion=dept,
                    confidence=conf,
                    matched_evidence=evidence,
                    explanation=expl,
                ))
                trace.append(
                    f"[DEPT] '{rule_name}' → {dept} "
                    f"(conf={conf:.0%}, {len(matched_pats)} match(es): {evidence[:2]})"
                )

        # ── 2. Priority keyword rules ──────────────────────────
        prio_scores: Dict[str, float] = {}
        for rule_name, patterns, prio, base_conf, expl in PRIORITY_KEYWORD_RULES:
            matched_pats = [p for p in patterns if re.search(p, text_lower)]
            if matched_pats:
                conf = min(base_conf + 0.015 * (len(matched_pats) - 1), 0.99)
                prio_scores[prio] = max(prio_scores.get(prio, 0.0), conf)
                evidence = self._clean_patterns(matched_pats[:3])
                rules_fired.append(RuleFiring(
                    rule_name=rule_name,
                    rule_type="priority",
                    conclusion=prio,
                    confidence=conf,
                    matched_evidence=evidence,
                    explanation=expl,
                ))
                trace.append(
                    f"[PRIO] '{rule_name}' → {prio} "
                    f"(conf={conf:.0%}, {len(matched_pats)} match(es): {evidence[:2]})"
                )

        # ── 3. Override rules ──────────────────────────────────
        override_dept: Optional[str] = None
        override_dept_conf: float = 0.0
        override_prio: Optional[str] = None
        override_prio_conf: float = 0.0

        for rule in OVERRIDE_RULES:
            try:
                fired = rule["condition"](text_lower, metadata)
            except Exception:
                fired = False

            if fired:
                conf = rule["confidence"]
                rules_fired.append(RuleFiring(
                    rule_name=rule["name"],
                    rule_type="override",
                    conclusion=f"dept={rule['department']} prio={rule['priority']}",
                    confidence=conf,
                    matched_evidence=[],
                    explanation=rule["explanation"],
                ))
                trace.append(f"[OVERRIDE] '{rule['name']}' fired (conf={conf:.0%})")

                if rule["department"] and conf > override_dept_conf:
                    override_dept = rule["department"]
                    override_dept_conf = conf

                if rule["priority"] and conf > override_prio_conf:
                    # Never downgrade via override — keep the more severe level
                    cur_ord = PRIORITY_ORDER.get(override_prio or "low", 0)
                    new_ord = PRIORITY_ORDER.get(rule["priority"], 0)
                    if new_ord >= cur_ord:
                        override_prio = rule["priority"]
                        override_prio_conf = conf

        # ── 4. Resolve final department ────────────────────────
        logic_dept = (
            max(dept_scores, key=dept_scores.get) if dept_scores else None
        )
        logic_dept_conf = dept_scores.get(logic_dept, 0.0) if logic_dept else 0.0

        if override_dept and override_dept_conf >= self.override_threshold:
            final_dept = override_dept
            dept_conf = override_dept_conf
            dept_source = "logic_override"
        elif (
            logic_dept
            and logic_dept_conf >= self.override_threshold
            and ml_department
            and logic_dept != ml_department
        ):
            final_dept = logic_dept
            dept_conf = logic_dept_conf
            dept_source = "logic_override"
        elif ml_department:
            if logic_dept == ml_department:
                # Agreement boosts confidence
                dept_conf = min(0.75 + logic_dept_conf * 0.18, 0.99)
                dept_source = "ml+logic"
            else:
                dept_conf = 0.65
                dept_source = "ml"
            final_dept = ml_department
        elif logic_dept:
            final_dept = logic_dept
            dept_conf = logic_dept_conf
            dept_source = "logic_only"
        else:
            final_dept = "general"
            dept_conf = 0.40
            dept_source = "default"

        # ── 5. Resolve final priority ──────────────────────────
        # Pick logic priority with highest severity among high-scoring ones
        logic_prio: Optional[str] = None
        logic_prio_conf: float = 0.0
        if prio_scores:
            # Among scores ≥ 0.65, pick the most severe
            candidates = {p: s for p, s in prio_scores.items() if s >= 0.65}
            if candidates:
                logic_prio = max(candidates, key=lambda p: (candidates[p], PRIORITY_ORDER.get(p, 0)))
                logic_prio_conf = candidates[logic_prio]
            else:
                logic_prio = max(prio_scores, key=prio_scores.get)
                logic_prio_conf = prio_scores[logic_prio]

        if override_prio and override_prio_conf >= self.override_threshold:
            # Never downgrade below ML prediction
            if ml_priority and PRIORITY_ORDER.get(override_prio, 0) < PRIORITY_ORDER.get(ml_priority, 0):
                final_prio = ml_priority
                prio_conf = 0.75
                prio_source = "ml"
            else:
                final_prio = override_prio
                prio_conf = override_prio_conf
                prio_source = "logic_override"
        elif (
            logic_prio
            and logic_prio_conf >= self.override_threshold
            and ml_priority
            and logic_prio != ml_priority
            and PRIORITY_ORDER.get(logic_prio, 0) > PRIORITY_ORDER.get(ml_priority, 0)
        ):
            # Only override upward (never downgrade via keyword logic alone)
            final_prio = logic_prio
            prio_conf = logic_prio_conf
            prio_source = "logic_override"
        elif ml_priority:
            if logic_prio == ml_priority:
                prio_conf = min(0.75 + logic_prio_conf * 0.18, 0.99)
                prio_source = "ml+logic"
            else:
                prio_conf = 0.65
                prio_source = "ml"
            final_prio = ml_priority
        elif logic_prio:
            final_prio = logic_prio
            prio_conf = logic_prio_conf
            prio_source = "logic_only"
        else:
            final_prio = "medium"
            prio_conf = 0.40
            prio_source = "default"

        # ── 6. Agreement & human-review flags ─────────────────
        dept_agreement = (final_dept == ml_department) if ml_department else True
        prio_agreement = (final_prio == ml_priority) if ml_priority else True

        needs_review = False
        review_reason = ""
        if self.review_on_disagreement:
            if not dept_agreement and dept_conf > 0.85 and dept_source == "logic_override":
                needs_review = True
                review_reason += (
                    f"Dept conflict: ML={ml_department}, Logic={logic_dept}. "
                )
            if not prio_agreement and prio_conf > 0.85 and prio_source == "logic_override":
                needs_review = True
                review_reason += (
                    f"Priority escalated: ML={ml_priority} → {final_prio}. "
                )

        trace.append(
            f"[FINAL] Department={final_dept} "
            f"(source={dept_source}, conf={dept_conf:.0%})"
        )
        trace.append(
            f"[FINAL] Priority={final_prio} "
            f"(source={prio_source}, conf={prio_conf:.0%})"
        )
        if needs_review:
            trace.append(f"[FLAG] Human review needed: {review_reason.strip()}")

        return ExplanationResult(
            ml_department=ml_department,
            ml_priority=ml_priority,
            final_department=final_dept,
            final_priority=final_prio,
            dept_source=dept_source,
            priority_source=prio_source,
            dept_confidence=dept_conf,
            priority_confidence=prio_conf,
            rules_fired=rules_fired,
            reasoning_trace=trace,
            dept_agreement=dept_agreement,
            priority_agreement=prio_agreement,
            needs_human_review=needs_review,
            review_reason=review_reason.strip(),
        )

    def explain_batch(
        self,
        texts: List[str],
        ml_departments: Optional[List[str]] = None,
        ml_priorities: Optional[List[str]] = None,
        metadata_list: Optional[List[Dict]] = None,
    ) -> List[ExplanationResult]:
        """Apply explain() to a list of tickets."""
        n = len(texts)
        ml_depts = ml_departments or [None] * n
        ml_prios = ml_priorities or [None] * n
        metas = metadata_list or [None] * n
        return [
            self.explain(t, d, p, m)
            for t, d, p, m in zip(texts, ml_depts, ml_prios, metas)
        ]

    # ── Reporting ──────────────────────────────────────────────

    def generate_batch_report(self, results: List["ExplanationResult"]) -> str:
        """Generate a summary statistics report for a batch of explanations."""
        n = len(results)
        if n == 0:
            return "No results to report."

        dept_overrides = sum(1 for r in results if "override" in r.dept_source)
        prio_overrides = sum(1 for r in results if "override" in r.priority_source)
        reviews_needed = sum(1 for r in results if r.needs_human_review)
        ml_logic_agree = sum(1 for r in results if r.dept_agreement and r.priority_agreement)

        avg_dept_conf = sum(r.dept_confidence for r in results) / n
        avg_prio_conf = sum(r.priority_confidence for r in results) / n

        dept_dist: Dict[str, int] = {}
        prio_dist: Dict[str, int] = {}
        for r in results:
            dept_dist[r.final_department] = dept_dist.get(r.final_department, 0) + 1
            prio_dist[r.final_priority] = prio_dist.get(r.final_priority, 0) + 1

        lines = [
            "=" * 62,
            "  SmartTicket — Explainability Layer Batch Report",
            "=" * 62,
            f"  Tickets analysed         : {n}",
            f"  ML + Logic agreement     : {ml_logic_agree} ({ml_logic_agree/n:.0%})",
            f"  Department overrides     : {dept_overrides} ({dept_overrides/n:.0%})",
            f"  Priority overrides       : {prio_overrides} ({prio_overrides/n:.0%})",
            f"  Human review flagged     : {reviews_needed} ({reviews_needed/n:.0%})",
            f"  Avg dept confidence      : {avg_dept_conf:.0%}",
            f"  Avg priority confidence  : {avg_prio_conf:.0%}",
            "",
            "  Department Distribution:",
        ]
        for dept in DEPARTMENTS:
            count = dept_dist.get(dept, 0)
            bar_len = int(count / n * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            lines.append(f"    {dept:<12} {count:>4}  {bar}  {count/n:.0%}")
        lines.append("")
        lines.append("  Priority Distribution:")
        for prio in PRIORITIES:
            count = prio_dist.get(prio, 0)
            bar_len = int(count / n * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            lines.append(f"    {prio:<12} {count:>4}  {bar}  {count/n:.0%}")
        lines.append("=" * 62)
        return "\n".join(lines)

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _clean_patterns(patterns: List[str]) -> List[str]:
        """Convert regex patterns to readable evidence strings."""
        cleaned = []
        for p in patterns:
            c = re.sub(r"\\b|\\s\+?|\(\?[^)]*\)|[()\\?+*{}]", "", p)
            c = c.replace(".?", " ").strip()
            if c:
                cleaned.append(c)
        return cleaned or ["(matched)"]

    @staticmethod
    def print_interpretability_report() -> None:
        """Print the short report on how logic improves interpretability."""
        print("""
╔══════════════════════════════════════════════════════════════════╗
║      How Logic Improves Interpretability & Control               ║
║      SmartTicket — Explainability Layer Report (Week 8)          ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  PROBLEM WITH PURE ML                                            ║
║  ML classifiers (KNN / RF / SVM) are black boxes.  They          ║
║  produce accurate predictions but cannot explain *why*.          ║
║  Business stakeholders cannot audit decisions, and engineers     ║
║  cannot quickly patch known failure modes.                       ║
║                                                                  ║
║  1. TRANSPARENCY                                                 ║
║     Every classification ships with a full reasoning trace —     ║
║     which symbolic rules fired, what textual evidence matched,   ║
║     and the resulting confidence.  No ML internals needed.       ║
║                                                                  ║
║  2. CONTROL                                                      ║
║     Domain experts author, tune, or delete rules in minutes      ║
║     without re-training any model.  A known failure mode         ║
║     (e.g., "hacked" tickets routed to billing) is fixed with     ║
║     one rule edit, not a full retraining cycle.                  ║
║                                                                  ║
║  3. OVERRIDE & SAFETY-NETTING                                    ║
║     High-confidence rules (≥ 90%) override ML when they detect  ║
║     clear domain signals — e.g., "unauthorized access" forces    ║
║     account + urgent regardless of ML output.  Priority is       ║
║     only *escalated*, never downgraded, by logic.                ║
║                                                                  ║
║  4. HUMAN-IN-THE-LOOP FLAGGING                                   ║
║     When ML and logic strongly disagree, the ticket is flagged   ║
║     for human review.  This replaces silent misclassification    ║
║     with an auditable escalation pipeline.                       ║
║                                                                  ║
║  5. NEURO-SYMBOLIC SYNERGY                                       ║
║     ML + logic agreement boosts the confidence score.  High-     ║
║     agreement tickets are auto-routed; low-agreement tickets     ║
║     enter a review queue.  The best of both worlds.              ║
║                                                                  ║
║  DESIGN DECISIONS                                                ║
║  • Override threshold : 90% — prevents over-aggressive rules.    ║
║  • Priority escalation only — logic never lowers urgency.        ║
║  • Rule evidence stored — enables future rule refinement.        ║
║  • Confidence calibrated to reflect joint ML+logic agreement.    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


# ══════════════════════════════════════════════════════════════
# Rule Set Overview (for documentation / grading)
# ══════════════════════════════════════════════════════════════

RULE_CATALOGUE = {
    "department_rules": [
        {
            "name": r["name"] if isinstance(r, dict) else r[0],
            "targets": r["department"] if isinstance(r, dict) else r[2],
            "confidence": r["confidence"] if isinstance(r, dict) else r[3],
            "explanation": r["explanation"] if isinstance(r, dict) else r[4],
        }
        for r in [
            {"name": e[0], "department": e[2], "confidence": e[3], "explanation": e[4]}
            for e in DEPT_KEYWORD_RULES
        ]
    ],
    "priority_rules": [
        {
            "name": e[0],
            "targets": e[2],
            "confidence": e[3],
            "explanation": e[4],
        }
        for e in PRIORITY_KEYWORD_RULES
    ],
    "override_rules": [
        {
            "name": r["name"],
            "forces_dept": r["department"],
            "forces_priority": r["priority"],
            "confidence": r["confidence"],
            "explanation": r["explanation"],
        }
        for r in OVERRIDE_RULES
    ],
}


def print_rule_catalogue() -> None:
    """Print the full symbolic rule set in a readable table."""
    print("\n" + "=" * 60)
    print("  SmartTicket — Symbolic Rule Catalogue")
    print("=" * 60)

    print("\n── Department Rules ─────────────────────────────────────")
    for r in RULE_CATALOGUE["department_rules"]:
        print(f"  [{r['confidence']:.0%}] {r['name']:<28} → {r['targets']}")
        print(f"         {r['explanation']}")

    print("\n── Priority Rules ───────────────────────────────────────")
    for r in RULE_CATALOGUE["priority_rules"]:
        print(f"  [{r['confidence']:.0%}] {r['name']:<28} → {r['targets']}")
        print(f"         {r['explanation']}")

    print("\n── Override Rules (fire before ML comparison) ───────────")
    for r in RULE_CATALOGUE["override_rules"]:
        dept_str = r["forces_dept"] or "unchanged"
        prio_str = r["forces_priority"] or "unchanged"
        print(f"  [{r['confidence']:.0%}] {r['name']}")
        print(f"         Forces dept={dept_str}, priority={prio_str}")
        print(f"         {r['explanation']}")
    print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════
# Demo / Script Entry Point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    explainer = TicketExplainer()

    # Print interpretability report
    explainer.print_interpretability_report()

    # Print rule catalogue
    print_rule_catalogue()

    # Example explanations
    EXAMPLES = [
        {
            "text": "I was charged twice for order #45231. The duplicate charge of $49.99 appeared on my credit card. Please refund ASAP.",
            "ml_department": "billing",
            "ml_priority": "medium",
            "metadata": {},
        },
        {
            "text": "My account has been hacked! Someone changed my email and I can't log in anymore. This is urgent please help immediately!",
            "ml_department": "technical",   # intentionally wrong — override should fix
            "ml_priority": "medium",
            "metadata": {},
        },
        {
            "text": "The app keeps crashing on my iPhone with error code ERR-5432. Tried reinstalling three times.",
            "ml_department": "technical",
            "ml_priority": "low",
            "metadata": {"escalated": 1},  # escalation override should raise priority
        },
        {
            "text": "My order #90123 was supposed to arrive 5 days ago but it never arrived. The tracking number shows delivered but I never received it.",
            "ml_department": "shipping",
            "ml_priority": "medium",
            "metadata": {"num_replies": 6},
        },
        {
            "text": "Hi, just wondering if you offer gift wrapping? Also do you ship to Australia? Thanks!",
            "ml_department": "general",
            "ml_priority": "low",
            "metadata": {},
        },
    ]

    print("\n" + "=" * 60)
    print("  Example Explanations")
    print("=" * 60)

    for i, ex in enumerate(EXAMPLES, 1):
        print(f"\n─── Ticket {i} ─────────────────────────────────────────────")
        print(f"Text: {ex['text'][:90]}...")
        print(f"ML predicted: dept={ex['ml_department']}, priority={ex['ml_priority']}")
        print()

        result = explainer.explain(
            text=ex["text"],
            ml_department=ex["ml_department"],
            ml_priority=ex["ml_priority"],
            metadata=ex["metadata"],
        )

        print(result.summary())
        print()
        print("  Reasoning trace:")
        for step in result.reasoning_trace:
            print(f"    {step}")

    # Batch report demo
    texts = [ex["text"] for ex in EXAMPLES]
    ml_depts = [ex["ml_department"] for ex in EXAMPLES]
    ml_prios = [ex["ml_priority"] for ex in EXAMPLES]
    metas = [ex["metadata"] for ex in EXAMPLES]

    batch_results = explainer.explain_batch(texts, ml_depts, ml_prios, metas)
    print("\n" + explainer.generate_batch_report(batch_results))
