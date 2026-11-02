"""Heuristic prompt-injection detector (runs on the raw ticket text in `normalize`).

Detection only flags; the flag forces escalation in routing (code decides, not the model).
Each pattern belongs to a family so tests and reports can show *why* a ticket was flagged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_I = re.IGNORECASE | re.MULTILINE

PATTERNS: dict[str, list[re.Pattern[str]]] = {
    # Fake role / authority headers: "SYSTEM:", "[ADMIN]", "<system>", "### Assistant"
    "role_spoof": [
        re.compile(
            r"^\s*[\[\(<#*\s]*(system|admin(istrator)?|assistant|developer|operator|"
            r"kestrel\s+(staff|admin|internal))\s*[\]\)>:*]+",
            _I,
        ),
        re.compile(r"</?\s*(system|instructions?|admin)\s*>", _I),
        re.compile(r"\[/?INST\]|<\|im_(start|end)\|>|<\|system\|>", _I),
    ],
    # "Ignore previous instructions", "you are now", "new instructions"
    "instruction_override": [
        re.compile(
            r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(previous|prior|above|earlier|"
            r"all|your|the)\b[^.\n]{0,20}\b(instructions?|rules|prompts?|guidelines|polic(y|ies))",
            _I,
        ),
        re.compile(r"\byou are (now|no longer)\b", _I),
        re.compile(r"\bnew (instructions?|rules)\s*:", _I),
        re.compile(r"\b(act|behave|respond) as (an?|the) (admin|developer|system)\b", _I),
    ],
    # Asking for the system prompt / internal rules
    "prompt_exfiltration": [
        re.compile(
            r"\b(reveal|show|print|repeat|output|tell me|share|display|leak)\b[^.\n]{0,40}"
            r"\b(system prompt|your (instructions|prompt|rules|guidelines)|internal "
            r"(rules|instructions|polic(y|ies)|notes)|hidden (instructions|prompt))",
            _I,
        ),
        re.compile(r"\bwhat (are|were) your (instructions|rules|system prompt)\b", _I),
    ],
    # Claims that policy changed / staff authorized something
    "policy_spoof": [
        re.compile(r"\bpolicy (has been |was |is )?(updated|changed|overridden)\b", _I),
        re.compile(
            r"\b(agents?|assistants?|bots?|the ai|you) (must|are required to|are instructed to) "
            r"(now |immediately )?(refund|issue|approve|ignore|send|credit|reveal|disclose)\b",
            _I,
        ),
        re.compile(
            r"\b(pre-?)?(authori[sz]ed|approved) by (a |the )?(manager|management|admin|"
            r"supervisor|kestrel|support lead|ceo)\b",
            _I,
        ),
        re.compile(r"\b(override|authori[sz]ation|approval) code\b", _I),
    ],
    # Directly coercing tools / bypassing review
    "tool_coercion": [
        re.compile(r"\b(issue_refund|lookup_order|get_customer_orders|search_kb)\b", _I),
        re.compile(r"\b(call|invoke|run|execute|use) (the )?\w*\s?(tool|function)\b", _I),
        re.compile(r"\bwithout (any )?(human )?(review|approval|verification|checks?)\b", _I),
        re.compile(r"\b(bypass|skip) (the )?(review|approval|limit|verification|checks?)\b", _I),
        re.compile(
            r"\b(refund|credit|transfer|send|wire)\b[^.\n]{0,40}\$\s?(\d{1,3}(,\d{3})+|\d{4,})",
            _I,
        ),
    ],
    # Hidden content: HTML comments, zero-width characters, long base64 blobs
    "hidden_content": [
        re.compile(r"<!--.*?-->", re.DOTALL),
        re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]"),
        re.compile(r"\b[A-Za-z0-9+/]{60,}={0,2}"),
        re.compile(r"\bdisplay\s*:\s*none\b|font-size\s*:\s*0", _I),
    ],
}


@dataclass(frozen=True)
class InjectionResult:
    flag: bool
    reasons: list[str]


def detect(text: str) -> InjectionResult:
    reasons: list[str] = []
    for family, patterns in PATTERNS.items():
        for pat in patterns:
            m = pat.search(text)
            if m:
                snippet = m.group(0).strip().replace("\n", " ")[:60]
                reasons.append(f"{family}: {snippet!r}")
                break  # one reason per family is enough
    return InjectionResult(flag=bool(reasons), reasons=reasons)
