import json
import math

import numpy as np
import pytest

from llikert.server.numerics import NonFiniteLogits, score_candidates


def test_hand_computed_small_vocabulary():
    # weights 4, 2, 1, 1 -> Z = 8; candidates 0 and 1 have q = 1/2 and 1/4
    logits = np.log([4.0, 2.0, 1.0, 1.0]) + 7.25  # a constant shift must not matter
    s = score_candidates(logits, [0, 1], values=[1.0, 5.0])
    assert s.candidate_probs == pytest.approx([0.5, 0.25], abs=1e-15)
    assert s.coverage == pytest.approx(0.75, abs=1e-15)
    assert s.log_coverage == pytest.approx(math.log(0.75), abs=1e-15)
    assert s.probabilities == pytest.approx([2 / 3, 1 / 3], abs=1e-15)
    assert s.expected_value == pytest.approx(7 / 3, abs=1e-14)
    assert s.candidate_log_probs == pytest.approx([math.log(0.5), math.log(0.25)], abs=1e-15)


def test_order_of_candidates_is_preserved():
    logits = np.log([4.0, 2.0, 1.0, 1.0])
    s = score_candidates(logits, [1, 0])
    assert s.probabilities == pytest.approx([1 / 3, 2 / 3], abs=1e-15)


def test_underflow_keeps_finite_log_values():
    logits = np.zeros(1000)
    logits[1], logits[2] = -800.0, -801.0
    s = score_candidates(logits, [1, 2])
    assert s.candidate_probs == [0.0, 0.0]
    assert s.coverage == 0.0
    assert all(math.isfinite(v) for v in s.candidate_log_probs)
    assert math.isfinite(s.log_coverage)
    assert s.candidate_log_probs[0] == pytest.approx(-800.0 - math.log(998 + math.exp(-800) + math.exp(-801)), abs=1e-9)
    assert s.probabilities == pytest.approx([1 / (1 + math.exp(-1)), math.exp(-1) / (1 + math.exp(-1))], abs=1e-12)
    assert sum(s.probabilities) == pytest.approx(1.0, abs=1e-15)


def test_low_coverage_concentrated_distribution():
    logits = np.zeros(50_000)
    logits[0], logits[1] = -5.0, -20.0
    s = score_candidates(logits, [0, 1])
    assert s.coverage < 1e-6
    assert s.probabilities[0] > 0.999999


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_logits_are_errors(bad):
    logits = np.zeros(10)
    logits[7] = bad
    with pytest.raises(NonFiniteLogits):
        score_candidates(logits, [0, 1])


def test_no_expected_value_without_values():
    assert score_candidates(np.zeros(4), [0, 1]).expected_value is None


def test_negative_and_noninteger_values():
    s = score_candidates(np.log([1.0, 3.0]), [0, 1], values=[-1.5, 0.25])
    assert s.expected_value == pytest.approx(0.25 * -1.5 + 0.75 * 0.25, abs=1e-15)


def test_float32_input_is_reduced_in_float64():
    logits32 = np.array([30.0, 29.999998, 0.0], dtype=np.float32)
    s = score_candidates(logits32, [0, 1])
    assert s.probabilities[0] != s.probabilities[1]


def test_shared_numeric_fixture(fixtures):
    for case in json.loads((fixtures / "numerics" / "cases.json").read_text())["cases"]:
        s = score_candidates(np.array(case["logits"]), case["token_ids"], case.get("values"))
        exp = case["expected"]
        assert s.probabilities == pytest.approx(exp["probabilities"], abs=1e-12), case["name"]
        assert s.candidate_log_probs == pytest.approx(exp["candidate_log_probs"], abs=1e-12), case["name"]
        assert s.log_coverage == pytest.approx(exp["log_coverage"], abs=1e-12), case["name"]
        assert s.expected_value == pytest.approx(exp["expected_value"], abs=1e-12) if exp["expected_value"] is not None else s.expected_value is None
