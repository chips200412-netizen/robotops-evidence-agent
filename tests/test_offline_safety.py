"""标准库测试：独立断言安全边界，不使用原工程业务资料。"""

import copy
import io
import itertools
import json
import unittest
from contextlib import redirect_stdout

from app.agent.execution import claim_tool_iteration
from app.agent.safety import (
    LLM_FAILURE_NOTE,
    RISK_FAILURE_NOTE,
    fail_closed_state,
    human_guidance,
    is_risk_level,
    merge_risk_levels,
    needs_human,
    safety_handoff,
)
from examples.offline_safety_demo import main, run_scenarios


class SafetyContractTests(unittest.TestCase):
    def test_risk_levels_use_exact_enum(self):
        for value in ("低", "中", "高"):
            with self.subTest(value=value):
                self.assertTrue(is_risk_level(value))
        for value in ("", "低 ", "HIGH", None, 1, True, [], {}):
            with self.subTest(value=value):
                self.assertFalse(is_risk_level(value))

    def test_risk_never_decreases_regardless_of_order(self):
        for levels in itertools.permutations(("低", "中", "高")):
            with self.subTest(levels=levels):
                self.assertEqual(merge_risk_levels(*levels), "高")
        self.assertEqual(merge_risk_levels("中", "低", ""), "中")
        self.assertEqual(merge_risk_levels("", "低", ""), "低")
        self.assertEqual(merge_risk_levels("", ""), "")

    def test_invalid_risk_is_rejected_even_after_high_risk(self):
        for value in ("未知", "低 ", None, 1, True, [], {}):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    merge_risk_levels("高", value)

    def test_each_safety_signal_requires_human(self):
        for signal in (
            {"risk_level": "高"}, {"urgency": "高"},
            {"need_human": True}, {"risk_check_failed": True},
            {"llm_failed": True},
        ):
            with self.subTest(signal=signal):
                self.assertTrue(needs_human(signal))
                self.assertTrue(safety_handoff(signal)["need_human"])

    def test_routine_state_has_no_safety_override(self):
        state = {"risk_level": "低", "urgency": "低", "need_human": False}
        self.assertFalse(needs_human(state))
        self.assertEqual(safety_handoff(state), {})

    def test_failure_flags_require_boolean_true(self):
        for value in (False, 1, "true", None):
            with self.subTest(value=value):
                self.assertFalse(needs_human({
                    "need_human": value, "risk_check_failed": value,
                    "llm_failed": value,
                }))

    def test_risk_failure_overrides_stale_diagnosis(self):
        state = {
            "risk_level": "低", "risk_check_failed": True,
            "diagnosis": ["合成旧结论"], "solution_steps": ["合成旧建议"],
        }
        original = copy.deepcopy(state)
        result = {**state, **safety_handoff(state)}
        self.assertEqual(state, original)
        self.assertEqual(result["risk_level"], "高")
        self.assertTrue(result["need_human"])
        self.assertEqual(result["risk_note"], RISK_FAILURE_NOTE)
        self.assertNotIn("合成旧结论", result["diagnosis"])
        self.assertNotIn("合成旧建议", result["solution_steps"])

    def test_model_failure_is_distinct_and_keeps_both_failure_flags(self):
        result = safety_handoff({
            "risk_check_failed": True, "llm_failed": True,
            "llm_failure_node": "synthetic_generate",
        })
        self.assertTrue(result["risk_check_failed"])
        self.assertTrue(result["llm_failed"])
        self.assertTrue(result["need_human"])
        self.assertEqual(result["risk_level"], "高")
        self.assertEqual(result["risk_note"], RISK_FAILURE_NOTE)
        self.assertEqual(result["diagnosis"], [LLM_FAILURE_NOTE])
        self.assertEqual(result["llm_failure_node"], "synthetic_generate")

    def test_guidance_and_failure_lists_are_not_shared(self):
        first = human_guidance("高")
        second = human_guidance("高")
        first["solution_steps"].clear()
        self.assertEqual(len(second["solution_steps"]), 3)
        first_failure = fail_closed_state()
        second_failure = fail_closed_state()
        first_failure["diagnosis"].clear()
        self.assertEqual(len(second_failure["diagnosis"]), 1)


class ToolBudgetTests(unittest.TestCase):
    def test_four_claims_allowed_then_fifth_denied(self):
        state = {"tool_iterations": 0, "max_tool_iterations": 4}
        for expected_count in (1, 2, 3, 4):
            before = dict(state)
            claim = claim_tool_iteration(state)
            self.assertEqual(state, before)
            self.assertTrue(claim.allowed)
            state.update(claim.update)
            self.assertEqual(state["tool_iterations"], expected_count)
        denied = claim_tool_iteration(state)
        self.assertFalse(denied.allowed)
        self.assertEqual(denied.update["tool_iterations"], 4)
        self.assertTrue(denied.update["need_human"])
        self.assertEqual(denied.update["execution_stop_reason"], "MAX_TOOL_ITERATIONS")
        state.update(denied.update)
        self.assertFalse(claim_tool_iteration(state).allowed)

    def test_corrupt_or_overridden_budget_fails_closed(self):
        for state in (
            {"tool_iterations": -1}, {"tool_iterations": True},
            {"tool_iterations": "0"}, {"tool_iterations": 5},
            {"tool_iterations": 0, "max_tool_iterations": 5},
            {"tool_iterations": 0, "max_tool_iterations": 4.0},
            {"tool_iterations": 0, "max_tool_iterations": True},
        ):
            with self.subTest(state=state):
                claim = claim_tool_iteration(state)
                self.assertFalse(claim.allowed)
                self.assertTrue(claim.update["need_human"])
                self.assertEqual(claim.update["max_tool_iterations"], 4)


class OfflineDemoTests(unittest.TestCase):
    def test_all_synthetic_scenarios_are_repeatable(self):
        scenarios = run_scenarios()
        self.assertEqual(scenarios, run_scenarios())
        self.assertEqual(len(scenarios), 5)
        by_id = {scenario["id"]: scenario for scenario in scenarios}
        self.assertEqual(by_id["risk_monotonic"]["result"], "高")
        self.assertTrue(by_id["human_handoff"]["needs_human"])
        failures = by_id["failure_closed"]["variants"]
        self.assertEqual(len(failures), 2)
        for failure in failures:
            self.assertEqual(failure["result"]["risk_level"], "高")
            self.assertTrue(failure["result"]["need_human"])
        self.assertTrue(by_id["invalid_risk"]["result"]["rejected"])
        self.assertEqual(
            [item["allowed"] for item in by_id["tool_budget"]["decisions"]],
            [True, True, True, True, False],
        )
        self.assertEqual(by_id["tool_budget"]["actual_tool_calls"], 0)

    def test_cli_labels_synthetic_output_and_pending_metrics(self):
        output = io.StringIO()
        with redirect_stdout(output):
            main()
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["mode"], "SYNTHETIC_OFFLINE_DEMO")
        self.assertEqual(payload["metric_status"], "12_FORMAL_METRICS_PENDING")
        self.assertEqual(len(payload["scenarios"]), 5)


if __name__ == "__main__":
    unittest.main()
