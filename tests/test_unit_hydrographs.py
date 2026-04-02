"""Tests for GR6J unit hydrograph utilities."""

import numpy as np
import pytest

from wat_mod_giz.unit_hydrographs import NH, D, compute_uh_ordinates, convolve_uh


class TestComputeUhOrdinates:
    """Tests for UH ordinate generation."""

    def test_returns_expected_shapes(self) -> None:
        uh1, uh2 = compute_uh_ordinates(1.7)

        assert uh1.shape == (NH,)
        assert uh2.shape == (2 * NH,)

    def test_ordinates_sum_to_one(self) -> None:
        uh1, uh2 = compute_uh_ordinates(1.7)

        assert uh1.sum() == pytest.approx(1.0)
        assert uh2.sum() == pytest.approx(1.0)

    def test_first_ordinate_matches_s_curve_difference(self) -> None:
        x4 = 2.0
        uh1, uh2 = compute_uh_ordinates(x4)

        assert uh1[0] == pytest.approx((1.0 / x4) ** D)
        assert uh2[0] == pytest.approx(0.5 * (1.0 / x4) ** D)

    def test_ordinates_are_non_negative(self) -> None:
        uh1, uh2 = compute_uh_ordinates(3.5)

        assert np.all(uh1 >= 0.0)
        assert np.all(uh2 >= 0.0)


class TestConvolveUh:
    """Tests for one-timestep UH convolution."""

    def test_returns_first_state_as_output(self) -> None:
        states = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        ordinates = np.array([0.2, 0.3, 0.5], dtype=np.float64)

        _, output = convolve_uh(states, 4.0, ordinates)

        assert output == 1.0

    def test_shifts_existing_states_and_adds_new_input(self) -> None:
        states = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        ordinates = np.array([0.2, 0.3, 0.5], dtype=np.float64)

        new_states, _ = convolve_uh(states, 10.0, ordinates)

        np.testing.assert_allclose(new_states, np.array([4.0, 6.0, 5.0]))

    def test_zero_input_only_shifts_states(self) -> None:
        states = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        ordinates = np.array([0.2, 0.3, 0.5], dtype=np.float64)

        new_states, output = convolve_uh(states, 0.0, ordinates)

        assert output == 1.0
        np.testing.assert_allclose(new_states, np.array([2.0, 3.0, 0.0]))
