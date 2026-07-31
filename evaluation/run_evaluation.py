"""
Simple evaluation harness for the customer support agent.

Runs the same run_support_workflow() function used by the API and UI
against a fixed set of test cases (evaluation/test_cases.json) and
reports whether the intent, escalation flag, tool type, and evidence
source match what's expected. This is NOT a replacement for the pytest
suite in tests/ -- it's a quick, readable regression baseline: a way to
eyeball how the agent behaves across a spread of question types, and to
notice at a glance if a future rule change shifts something unexpected.

Run with:
    python evaluation/run_evaluation.py
"""

import json
import sys
from pathlib import Path

# Make the project root importable, since running this file directly only
# puts its own folder (evaluation/) on sys.path by default.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.graph.graph import run_support_workflow  # noqa: E402

TEST_CASES_PATH = Path(__file__).resolve().parent / "test_cases.json"


def load_test_cases() -> list[dict]:
    with open(TEST_CASES_PATH, encoding="utf-8") as f:
        return json.load(f)


def tool_type_for(intent: str, tool_result: dict | None) -> str | None:
    """Roughly categorize tool_result's shape, to compare against expected_tool_type."""
    if not tool_result:
        return None
    if intent == "order_status" and "status" in tool_result:
        return "order_lookup"
    if intent == "refund_request" and "eligible" in tool_result:
        return "refund_check"
    return None


def run_evaluation() -> None:
    test_cases = load_test_cases()
    passed_count = 0

    for case in test_cases:
        question = case["question"]
        result = run_support_workflow(question)

        actual_intent = result.get("intent")
        actual_escalation = result.get("needs_escalation", False)
        actual_tool_type = tool_type_for(actual_intent, result.get("tool_result"))
        actual_evidence_sources = [item["source"] for item in result.get("evidence") or []]

        intent_ok = actual_intent == case["expected_intent"]
        escalation_ok = actual_escalation == case["expected_needs_escalation"]

        expected_tool_type = case.get("expected_tool_type")
        tool_ok = expected_tool_type is None or actual_tool_type == expected_tool_type

        expected_evidence_source = case.get("expected_evidence_source")
        evidence_ok = (
            expected_evidence_source is None
            or expected_evidence_source in actual_evidence_sources
        )

        case_passed = intent_ok and escalation_ok and tool_ok and evidence_ok
        passed_count += int(case_passed)

        print(f"[{'PASS' if case_passed else 'FAIL'}] {question}")
        print(f"    expected_intent={case['expected_intent']!r}  actual_intent={actual_intent!r}")
        print(
            f"    expected_escalation={case['expected_needs_escalation']!r}  "
            f"actual_escalation={actual_escalation!r}"
        )
        if expected_tool_type is not None:
            print(f"    expected_tool_type={expected_tool_type!r}  actual_tool_type={actual_tool_type!r}")
        if expected_evidence_source is not None:
            print(
                f"    expected_evidence_source={expected_evidence_source!r}  "
                f"actual_evidence_sources={actual_evidence_sources!r}"
            )
        print()

    total = len(test_cases)
    accuracy = (passed_count / total * 100) if total else 0.0
    print("=" * 60)
    print(f"Summary: {passed_count}/{total} passed ({accuracy:.1f}% accuracy)")


if __name__ == "__main__":
    run_evaluation()
