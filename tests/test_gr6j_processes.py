"""Tests for pure GR6J process equations."""

import numpy as np
import pytest

from wat_mod_giz.processes.gr6j import (
    direct_branch,
    exponential_store_update,
    groundwater_exchange,
    percolation,
    production_store_update,
    routing_store_update,
)


class TestProductionStoreUpdate:
    def test_rainfall_dominant_case(self) -> None:
        new_store, actual_et, net_rainfall_pn, effective_rainfall_pr = production_store_update(20.0, 5.0, 100.0, 300.0)

        assert new_store > 100.0
        assert actual_et == 5.0
        assert net_rainfall_pn == 15.0
        assert 0.0 < effective_rainfall_pr < net_rainfall_pn

    def test_evapotranspiration_dominant_case(self) -> None:
        new_store, actual_et, net_rainfall_pn, effective_rainfall_pr = production_store_update(2.0, 10.0, 150.0, 300.0)

        assert new_store < 150.0
        assert actual_et > 2.0
        assert net_rainfall_pn == 0.0
        assert effective_rainfall_pr == 0.0


class TestPercolation:
    def test_zero_store_has_zero_percolation(self) -> None:
        new_store, perc = percolation(0.0, 300.0)
        assert new_store == 0.0
        assert perc == 0.0

    def test_percolation_is_bounded_by_store(self) -> None:
        new_store, perc = percolation(200.0, 300.0)
        assert 0.0 <= perc <= 200.0
        assert new_store + perc == pytest.approx(200.0)


class TestGroundwaterExchange:
    def test_exchange_matches_formula(self) -> None:
        exchange = groundwater_exchange(80.0, 2.0, 100.0, 0.5)
        assert exchange == pytest.approx(2.0 * (0.8 - 0.5))


class TestRoutingStoreUpdate:
    def test_store_cannot_go_negative(self) -> None:
        new_store, _, actual_exchange = routing_store_update(10.0, 5.0, -100.0, 100.0)
        assert new_store == pytest.approx(0.0)
        assert actual_exchange == pytest.approx(-15.0)


class TestExponentialStoreUpdate:
    def test_store_can_be_negative(self) -> None:
        new_store, qrexp = exponential_store_update(-5.0, 2.0, -3.0, 5.0)
        assert np.isfinite(new_store)
        assert np.isfinite(qrexp)


class TestDirectBranch:
    def test_negative_combined_flow_is_clamped(self) -> None:
        qd, actual_exchange = direct_branch(1.0, -5.0)
        assert qd == 0.0
        assert actual_exchange == -1.0
