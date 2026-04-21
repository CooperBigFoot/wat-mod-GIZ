"""Plotting helpers for the full bucket model notebook (08)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_state_panels(
    results_df: pd.DataFrame,
    soil_capacity_mm: float,
    year: int,
    output_path: Path | None = None,
) -> None:
    year_df = results_df.loc[results_df["date"].dt.year == year]

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(year_df["date"], year_df["snow_cover_mm"], color="tab:blue", linewidth=1.4)
    axes[0].set_ylabel("Snow [mm SWE]")
    axes[0].set_title(f"Alamedin {year}: simulated state variables")
    axes[0].grid(alpha=0.3)

    axes[1].plot(year_df["date"], year_df["soil_mm"], color="tab:green", linewidth=1.4)
    axes[1].axhline(soil_capacity_mm, color="tab:gray", linestyle="--", linewidth=1, label="soil capacity")
    axes[1].set_ylabel("Soil [mm]")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)

    axes[2].plot(year_df["date"], year_df["groundwater_mm"], color="tab:orange", linewidth=1.4)
    axes[2].set_ylabel("Groundwater [mm]")
    axes[2].set_xlabel("Date")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_hydrograph_comparison(
    results_df: pd.DataFrame,
    q_obs_df: pd.DataFrame,
    year: int,
    output_path: Path | None = None,
) -> None:
    year_sim = results_df.loc[results_df["date"].dt.year == year]
    year_obs = q_obs_df.loc[
        (q_obs_df["date"].dt.year == year) & (q_obs_df["q_status"] == "observed")
    ]

    fig, (ax_q, ax_p) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_q.plot(year_obs["date"], year_obs["q_mm_clean"], color="tab:blue", linewidth=1.2, label="Observed Q")
    ax_q.plot(
        year_sim["date"], year_sim["total_runoff_mm"],
        color="tab:orange", linewidth=1.2, label="Simulated Q (bucket model)",
    )
    ax_q.plot(
        year_sim["date"], year_sim["groundwater_runoff_mm"],
        color="tab:green", linewidth=1.0, linestyle="--", label="Simulated baseflow only",
    )
    ax_q.set_ylabel("Q [mm/day]")
    ax_q.set_title(f"Alamedin {year}: observed vs simulated streamflow")
    ax_q.legend(loc="upper right")
    ax_q.grid(alpha=0.3)

    ax_p.bar(year_sim["date"], year_sim["rain_mm"], color="tab:gray", width=1.0, label="Rain")
    ax_p.bar(
        year_sim["date"], year_sim["snowmelt_mm"],
        bottom=year_sim["rain_mm"], color="tab:cyan", width=1.0, label="Snowmelt",
    )
    ax_p.set_ylabel("Liquid input [mm/day]")
    ax_p.set_xlabel("Date")
    ax_p.legend(loc="upper right")
    ax_p.grid(axis="y", alpha=0.3)
    ax_p.invert_yaxis()

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_flow_duration_curves(
    results_df: pd.DataFrame,
    q_obs_df: pd.DataFrame,
    output_path: Path | None = None,
) -> None:
    merged = results_df.merge(q_obs_df[["date", "q_mm_clean"]], on="date", how="inner")
    obs = merged["q_mm_clean"].to_numpy()
    sim = merged["total_runoff_mm"].to_numpy()

    n = len(obs)
    exceedance_pct = np.arange(1, n + 1) / n * 100.0
    obs_sorted = np.sort(obs)[::-1]
    sim_sorted = np.sort(sim)[::-1]

    fig, (ax_low, ax_high) = plt.subplots(1, 2, figsize=(13, 5))

    ax_low.plot(exceedance_pct, obs_sorted, color="tab:blue", linewidth=1.4, label="Observed")
    ax_low.plot(exceedance_pct, sim_sorted, color="tab:orange", linewidth=1.4, label="Simulated")
    ax_low.set_yscale("log")
    ax_low.set_xlabel("Exceedance probability [%]")
    ax_low.set_ylabel("Q [mm/day] (log scale)")
    ax_low.set_title("Low-flow diagnosis: log Q vs linear exceedance")
    ax_low.grid(which="both", alpha=0.3)
    ax_low.legend(loc="upper right")

    ax_high.plot(exceedance_pct, obs_sorted, color="tab:blue", linewidth=1.4, label="Observed")
    ax_high.plot(exceedance_pct, sim_sorted, color="tab:orange", linewidth=1.4, label="Simulated")
    ax_high.set_xscale("log")
    ax_high.set_xlabel("Exceedance probability [%] (log scale)")
    ax_high.set_ylabel("Q [mm/day]")
    ax_high.set_title("High-flow diagnosis: linear Q vs log exceedance")
    ax_high.grid(which="both", alpha=0.3)
    ax_high.legend(loc="upper right")

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
