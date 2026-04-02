"""Pure GR6J process equations."""

from __future__ import annotations

import numpy as np

MAX_TANH_ARG: float = 13.0
MAX_EXP_ARG: float = 33.0
EXP_BRANCH_THRESHOLD: float = 7.0
PERC_CONSTANT: float = 25.62890625


def production_store_update(
    precip: float,
    pet: float,
    production_store: float,
    x1: float,
) -> tuple[float, float, float, float]:
    """Update the production store and return ET and effective rainfall terms."""
    store_ratio = production_store / x1

    if precip < pet:
        net_evap = pet - precip
        scaled_evap = min(net_evap / x1, MAX_TANH_ARG)
        tanh_ws = np.tanh(scaled_evap)

        numerator = (2.0 - store_ratio) * tanh_ws
        denominator = 1.0 + (1.0 - store_ratio) * tanh_ws
        evap_from_store = production_store * numerator / denominator

        actual_et = evap_from_store + precip
        new_store = production_store - evap_from_store
        return new_store, actual_et, 0.0, 0.0

    net_rainfall_pn = precip - pet
    actual_et = pet
    scaled_precip = min(net_rainfall_pn / x1, MAX_TANH_ARG)
    tanh_ws = np.tanh(scaled_precip)

    numerator = (1.0 - store_ratio**2) * tanh_ws
    denominator = 1.0 + store_ratio * tanh_ws
    storage_infiltration = x1 * numerator / denominator
    effective_rainfall_pr = net_rainfall_pn - storage_infiltration
    new_store = production_store + storage_infiltration

    return new_store, actual_et, net_rainfall_pn, effective_rainfall_pr


def percolation(production_store: float, x1: float) -> tuple[float, float]:
    """Compute production store percolation."""
    store = max(production_store, 0.0)
    store_ratio_4 = (store / x1) ** 4
    percolation_amount = store * (1.0 - (1.0 + store_ratio_4 / PERC_CONSTANT) ** (-0.25))
    return store - percolation_amount, percolation_amount


def groundwater_exchange(routing_store: float, x2: float, x3: float, x5: float) -> float:
    """Compute potential groundwater exchange."""
    return x2 * (routing_store / x3 - x5)


def routing_store_update(
    routing_store: float,
    uh1_output: float,
    exchange: float,
    x3: float,
) -> tuple[float, float, float]:
    """Update the routing store and compute routed outflow."""
    store_after_inflow = routing_store + uh1_output + exchange

    if store_after_inflow >= 0.0:
        actual_exchange = exchange
        store = store_after_inflow
    else:
        actual_exchange = -(routing_store + uh1_output)
        store = 0.0

    if store > 0.0:
        store_ratio_4 = (store / x3) ** 4
        outflow_qr = store * (1.0 - (1.0 + store_ratio_4) ** (-0.25))
    else:
        outflow_qr = 0.0

    return store - outflow_qr, outflow_qr, actual_exchange


def exponential_store_update(exp_store: float, uh1_output: float, exchange: float, x6: float) -> tuple[float, float]:
    """Update the exponential store and compute its outflow."""
    store = exp_store + uh1_output + exchange
    ar = max(-MAX_EXP_ARG, min(store / x6, MAX_EXP_ARG))

    if ar > EXP_BRANCH_THRESHOLD:
        outflow_qrexp = store + x6 / np.exp(ar)
    elif ar < -EXP_BRANCH_THRESHOLD:
        outflow_qrexp = x6 * np.exp(ar)
    else:
        outflow_qrexp = x6 * np.log(np.exp(ar) + 1.0)

    return store - outflow_qrexp, outflow_qrexp


def direct_branch(uh2_output: float, exchange: float) -> tuple[float, float]:
    """Compute direct branch outflow with non-negativity protection."""
    combined = uh2_output + exchange
    if combined >= 0.0:
        return combined, exchange
    return 0.0, -uh2_output
