from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "geomagnetic_mpl_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import special
from scipy.signal import find_peaks

MU0 = 4.0 * np.pi * 1e-7
KM_TO_M = 1000.0
NT_PER_T = 1e9
K2_MAX = 0.999999
EPS = 1e-30
BACKGROUND_COLOR = "#222222"
DOUBLE_COLOR = "#D55E00"

PARAMETERS = {'sq_source': {'N': 1000,
               'r_min_km': 200.0,
               'r_max_km': 3600.0,
               'I_inner_A': -160.0,
               'I_outer_A': -40.0,
               'z_E_km': -110.0,
               'y_s_km': 0.0,
               'x_st_km': -3125.0,
               'x_ed_km': 5875.0,
               'tau_1_h': 7.0,
               'tau_2_h': 20.0},
 'internal_source': {'z_I_km': 600.0, 'eta_I': -0.5, 'delta_tau_I_h': 1.0},
 'anomaly': {'x_a_km': 0.0,
             'y_a_km': 0.0,
             'z_a_km': 18.0,
             'lambda_in': 0.6,
             'I0_A': 1.0,
             'A_a_nT': 80.0},
 'time': {'tau_start_h': 0.0,
          'tau_end_h': 24.0,
          'tau_points': 721,
          'T_a': -30.0,
          'T_b': -5.0,
          'tau_h': 2.0,
          'tau_a_h': 3.75,
          'p': 1.5},
 'stations': {'names': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'],
              'x_station_km': [800.0, 0.0, -1200.0, 750.0, -750.0, 750.0],
              'y_station_km': [0.0, 800.0, 0.0, 750.0, -750.0, -900.0]},
 'grid': {'x_min_km': -2000.0,
          'x_max_km': 2000.0,
          'x_points': 161,
          'y_min_km': -2000.0,
          'y_max_km': 2000.0,
          'y_points': 161},
 'plot': {'typical_T': None}}


def tau_axis(config: dict[str, Any]) -> np.ndarray:
    t = config["time"]
    return np.linspace(float(t["tau_start_h"]), float(t["tau_end_h"]), int(t["tau_points"]))


def y_grid_km(config: dict[str, Any]) -> np.ndarray:
    g = config["grid"]
    return np.linspace(float(g["y_min_km"]), float(g["y_max_km"]), int(g["y_points"]))


def x_grid_km(config: dict[str, Any]) -> np.ndarray:
    g = config["grid"]
    return np.linspace(float(g["x_min_km"]), float(g["x_max_km"]), int(g["x_points"]))


def grid_xy_km(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(x_grid_km(config), y_grid_km(config), indexing="xy")


def station_names(config: dict[str, Any]) -> list[str]:
    return [str(name) for name in config["stations"]["names"]]


def x_station_km(config: dict[str, Any]) -> np.ndarray:
    return np.asarray(config["stations"]["x_station_km"], dtype=float)


def y_station_km(config: dict[str, Any]) -> np.ndarray:
    return np.asarray(config["stations"]["y_station_km"], dtype=float)


def station_data(config: dict[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = station_names(config)
    x = x_station_km(config)
    y = y_station_km(config)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Station x and y coordinates must be one-dimensional arrays.")
    if not len(names) == len(x) == len(y):
        raise ValueError("Station names, x coordinates, and y coordinates must have equal lengths.")
    return names, x, y


def typical_T(config: dict[str, Any]) -> float:
    value = config.get("plot", {}).get("typical_T")
    if value is not None:
        return float(value)
    return 0.5 * (float(config["time"]["T_a"]) + float(config["time"]["T_b"]))


def sq_rings(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    sq = config["sq_source"]
    n = int(sq["N"])
    radii_m = np.linspace(float(sq["r_min_km"]), float(sq["r_max_km"]), n) * KM_TO_M
    currents_A = np.linspace(float(sq["I_inner_A"]), float(sq["I_outer_A"]), n)
    return radii_m, currents_A


def K_z(r_m: Any, I_A: Any, rho_m: Any, delta_z_m: Any) -> np.ndarray:
    r = np.asarray(r_m, dtype=float)
    I = np.asarray(I_A, dtype=float)
    rho = np.abs(np.asarray(rho_m, dtype=float))
    dz = np.asarray(delta_z_m, dtype=float)

    beta2 = (r + rho) ** 2 + dz**2
    beta = np.sqrt(beta2 + EPS)
    alpha2 = np.maximum((r - rho) ** 2 + dz**2, EPS)
    k2 = 4.0 * r * rho / (beta2 + EPS)
    k2 = np.clip(k2, 0.0, K2_MAX)

    K = special.ellipk(k2)
    E = special.ellipe(k2)
    return NT_PER_T * (MU0 * I) / (2.0 * np.pi * beta) * (
        K + ((r**2 - rho**2 - dz**2) / alpha2) * E
    )


def _sum_rings(
    radii_m: np.ndarray,
    currents_A: np.ndarray,
    rho_m: Any,
    delta_z_m: Any,
    chunk_size: int = 16,
) -> np.ndarray:
    rho = np.asarray(rho_m, dtype=float)
    dz = np.asarray(delta_z_m, dtype=float)
    shape = np.broadcast_shapes(rho.shape, dz.shape)
    rho = np.broadcast_to(rho, shape)
    dz = np.broadcast_to(dz, shape)
    total = np.zeros(shape, dtype=float)

    for start in range(0, len(radii_m), chunk_size):
        stop = min(start + chunk_size, len(radii_m))
        ring_shape = (stop - start,) + (1,) * len(shape)
        total += K_z(
            radii_m[start:stop].reshape(ring_shape),
            currents_A[start:stop].reshape(ring_shape),
            rho,
            dz,
        ).sum(axis=0)
    return total


def x_s(tau_h: Any, config: dict[str, Any]) -> np.ndarray:
    sq = config["sq_source"]
    tau = np.mod(np.asarray(tau_h, dtype=float), 24.0)
    tau_1 = float(sq["tau_1_h"])
    tau_2 = float(sq["tau_2_h"])
    x_st = float(sq["x_st_km"]) * KM_TO_M
    x_ed = float(sq["x_ed_km"]) * KM_TO_M
    v_x = (x_ed - x_st) / (tau_2 - tau_1)
    moving = x_st + v_x * (tau - tau_1)
    return np.where(tau <= tau_1, x_st, np.where(tau >= tau_2, x_ed, moving))


def _tau_xy_arrays(tau_h: Any, x_km: Any, y_km: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = np.broadcast_arrays(np.asarray(x_km, dtype=float), np.asarray(y_km, dtype=float))
    spatial_shape = x.shape
    tau = np.atleast_1d(np.asarray(tau_h, dtype=float)).reshape((-1,) + (1,) * len(spatial_shape))
    return tau, x.reshape((1,) + spatial_shape), y.reshape((1,) + spatial_shape)


def Z_E(tau_h: Any, x_km: Any, y_km: Any, config: dict[str, Any]) -> np.ndarray:
    tau, x, y = _tau_xy_arrays(tau_h, x_km, y_km)
    sq = config["sq_source"]
    radii_m, currents_A = sq_rings(config)
    y_s = float(sq["y_s_km"]) * KM_TO_M
    rho = np.sqrt((x * KM_TO_M - x_s(tau, config)) ** 2 + (y * KM_TO_M - y_s) ** 2)
    dz = 0.0 - float(sq["z_E_km"]) * KM_TO_M
    return _sum_rings(radii_m, currents_A, rho, dz)


def Z_I(tau_h: Any, x_km: Any, y_km: Any, config: dict[str, Any]) -> np.ndarray:
    tau, x, y = _tau_xy_arrays(tau_h, x_km, y_km)
    sq = config["sq_source"]
    internal = config["internal_source"]
    radii_m, currents_A = sq_rings(config)
    y_s = float(sq["y_s_km"]) * KM_TO_M
    tau_I = tau - float(internal["delta_tau_I_h"])
    rho = np.sqrt((x * KM_TO_M - x_s(tau_I, config)) ** 2 + (y * KM_TO_M - y_s) ** 2)
    dz = 0.0 - float(internal["z_I_km"]) * KM_TO_M
    return _sum_rings(radii_m, float(internal["eta_I"]) * currents_A, rho, dz)


def Z_0(tau_h: Any, x_km: Any, y_km: Any, config: dict[str, Any]) -> np.ndarray:
    return Z_E(tau_h, x_km, y_km, config) + Z_I(tau_h, x_km, y_km, config)


def _raw_G_a(x_km: Any, y_km: Any, config: dict[str, Any]) -> np.ndarray:
    anomaly = config["anomaly"]
    radii_m, currents_A = sq_rings(config)
    x, y = np.broadcast_arrays(np.asarray(x_km, dtype=float), np.asarray(y_km, dtype=float))

    I0 = float(anomaly["I0_A"])
    lambda_in = float(anomaly["lambda_in"])
    weights = -lambda_in * currents_A / I0

    x_a = float(anomaly["x_a_km"]) * KM_TO_M
    y_a = float(anomaly["y_a_km"]) * KM_TO_M
    z_a = float(anomaly["z_a_km"]) * KM_TO_M
    rho = np.sqrt((x * KM_TO_M - x_a) ** 2 + (y * KM_TO_M - y_a) ** 2)
    dz = 0.0 - z_a

    total = np.zeros_like(rho, dtype=float)
    for radius, weight in zip(radii_m, weights):
        total += weight * K_z(radius, I0, rho, dz)
    return total


def G_a(
    x_km: Any,
    y_km: Any,
    config: dict[str, Any],
    normalize_x_km: Any | None = None,
    normalize_y_km: Any | None = None,
) -> np.ndarray:
    x, y = np.broadcast_arrays(np.asarray(x_km, dtype=float), np.asarray(y_km, dtype=float))
    raw = _raw_G_a(x, y, config)
    if normalize_x_km is None and normalize_y_km is None:
        x_norm, y_norm = grid_xy_km(config)
    elif normalize_x_km is None or normalize_y_km is None:
        raise ValueError("Both normalization coordinates must be provided together.")
    else:
        x_norm, y_norm = np.broadcast_arrays(
            np.asarray(normalize_x_km, dtype=float), np.asarray(normalize_y_km, dtype=float)
        )
    if raw.shape == x_norm.shape and np.array_equal(x, x_norm) and np.array_equal(y, y_norm):
        raw_norm = raw
    else:
        raw_norm = _raw_G_a(x_norm, y_norm, config)
    scale = float(np.nanmax(np.abs(raw_norm)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Cannot normalize G_a; raw kernel is zero or non-finite.")
    return raw / scale


def h_T(T: Any, config: dict[str, Any]) -> np.ndarray:
    T_arr = np.asarray(T, dtype=float)
    time = config["time"]
    tau_h = float(time["tau_h"])
    left = np.clip(-(T_arr - float(time["T_a"])) / tau_h, -700.0, 700.0)
    right = np.clip(-(T_arr - float(time["T_b"])) / tau_h, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(left)) - 1.0 / (1.0 + np.exp(right))


def _c_tau(tau_h: Any, config: dict[str, Any]) -> np.ndarray:
    tau = np.asarray(tau_h, dtype=float)
    sq = config["sq_source"]
    internal = config["internal_source"]
    anomaly = config["anomaly"]
    radii_m, currents_A = sq_rings(config)

    x_a = float(anomaly["x_a_km"]) * KM_TO_M
    y_a = float(anomaly["y_a_km"]) * KM_TO_M
    z_a = float(anomaly["z_a_km"]) * KM_TO_M
    y_s = float(sq["y_s_km"]) * KM_TO_M

    rho_E = np.sqrt((x_a - x_s(tau, config)) ** 2 + (y_a - y_s) ** 2)
    dz_E = z_a - float(sq["z_E_km"]) * KM_TO_M
    c_E = _sum_rings(radii_m, currents_A, rho_E, dz_E)

    tau_I = tau - float(internal["delta_tau_I_h"])
    rho_I = np.sqrt((x_a - x_s(tau_I, config)) ** 2 + (y_a - y_s) ** 2)
    dz_I = z_a - float(internal["z_I_km"]) * KM_TO_M
    c_I = _sum_rings(radii_m, float(internal["eta_I"]) * currents_A, rho_I, dz_I)
    return c_E + c_I


def a_tau(tau_h: Any, config: dict[str, Any]) -> np.ndarray:
    tau = np.asarray(tau_h, dtype=float)
    time = config["time"]
    shifted = np.mod(tau - float(time["tau_a_h"]), 24.0)
    c_shifted = _c_tau(shifted, config)
    scale = float(np.nanmax(np.abs(_c_tau(tau_axis(config), config))))
    d = c_shifted / scale
    p = float(time["p"])
    return np.sign(d) * np.abs(d) ** p


def format_hhmm(hour: float, *, allow_24: bool = False) -> str:
    total_minutes = int(np.floor(float(hour) * 60.0 + 0.5))
    if allow_24 and total_minutes == 24 * 60:
        return "24:00"
    total_minutes %= 24 * 60
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours:02d}:{minutes:02d}"


def set_clock_xticks(ax, ticks: np.ndarray) -> None:
    ax.set_xticks(
        ticks,
        [format_hhmm(value, allow_24=True) for value in ticks],
    )


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 7.6,
            "axes.titlesize": 8.4,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig, output_dir: Path, stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.03,
    )


def _two_low_points(tau_h: np.ndarray, values: np.ndarray) -> tuple[float, ...]:
    mask = (tau_h >= 7.0) & (tau_h <= 19.0)
    indices = np.flatnonzero(mask)
    step_h = float(np.median(np.diff(tau_h)))
    minimum_distance = max(1, int(round(0.75 / step_h)))
    peaks, _ = find_peaks(
        -values[indices],
        prominence=0.5,
        distance=minimum_distance,
    )
    candidates = indices[peaks]
    if len(candidates) < 2:
        raise ValueError("Double-minimum case did not produce two resolvable low points.")

    selected = sorted(candidates, key=lambda index: values[index])[:2]
    first_index, second_index = sorted(selected)
    first_time = float(tau_h[first_index])
    second_time = float(tau_h[second_index])
    return (
        first_time,
        float(values[first_index]),
        second_time,
        float(values[second_index]),
        second_time - first_time,
    )


def _station_title(panel: str, name: str, x: float, y: float) -> str:
    return rf"{panel}  ${name[0]}_{{{name[1:]}}}$: ({x:.0f}, {y:.0f}) km"


def _set_curve_axes(ax, metadata: dict[str, Any], station_index: int, panel: str) -> None:
    ax.axhline(0.0, color="0.42", lw=0.55, ls=":", zorder=0)
    ax.set_xlim(0.0, 24.0)
    set_clock_xticks(ax, np.arange(0.0, 25.0, 4.0))


    ax.tick_params(axis="x", labelbottom=True, labelsize=5.8, pad=1.0)
    ax.grid(True, color="0.88", lw=0.42)
    ax.set_axisbelow(True)
    ax.set_title(
        _station_title(
            panel,
            metadata["station_names"][station_index],
            metadata["x_station_km"][station_index],
            metadata["y_station_km"][station_index],
        ),
        pad=3.5,
    )


def _set_shared_limits(axes, curves: list[np.ndarray]) -> None:
    y_min = min(float(np.min(curve)) for curve in curves)
    y_max = max(float(np.max(curve)) for curve in curves)
    span = y_max - y_min
    for ax in axes.ravel():
        ax.set_ylim(y_min - 0.18 * span, y_max + 0.04 * span)


def _metric_box(ax, text: str) -> None:
    ax.text(
        0.018,
        0.025,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.05,
        linespacing=1.1,
        bbox={"facecolor": "white", "edgecolor": "0.78", "alpha": 0.9, "pad": 1.3},
        zorder=8,
    )


def compute_double_minimum(config: dict[str, Any]) -> tuple[dict, dict]:
    tau = tau_axis(config)
    tau = tau[tau < 24.0]
    names, x, y = station_data(config)
    background = Z_0(tau, x, y, config)
    activation = float(h_T([typical_T(config)], config)[0])
    modulation = a_tau(tau, config)
    kernel = G_a(x, y, config)
    anomaly = float(config["anomaly"]["A_a_nT"]) * activation * modulation[:, None] * kernel[None, :]
    total = background + anomaly
    minima = np.asarray([_two_low_points(tau, total[:, index]) for index in range(len(names))])
    metadata = {
        "tau_h": tau,
        "station_names": names,
        "x_station_km": x,
        "y_station_km": y,
        "background_nT": background,
        "background_min_time_h": tau[np.argmin(background, axis=0)],
    }
    result = {
        "total_nT": total,
        "anomaly_nT": anomaly,
        "first_low_time_h": minima[:, 0],
        "first_low_nT": minima[:, 1],
        "second_low_time_h": minima[:, 2],
        "second_low_nT": minima[:, 3],
        "low_gap_h": minima[:, 4],
        "total_range_nT": np.ptp(total, axis=0),
        "max_abs_anomaly_nT": np.max(np.abs(anomaly), axis=0),
    }
    return metadata, result


def plot_double_minimum(
    metadata: dict[str, Any], result: dict, config: dict, output_dir: Path, dpi: int
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.25), sharex=True, sharey=True)
    panels = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")
    for station_index, (ax, panel) in enumerate(zip(axes.ravel(), panels)):
        ax.plot(
            metadata["tau_h"],
            metadata["background_nT"][:, station_index],
            color=BACKGROUND_COLOR,
            lw=1.25,
            label="Background",
        )
        ax.plot(
            metadata["tau_h"],
            result["total_nT"][:, station_index],
            color=DOUBLE_COLOR,
            ls="--",
            lw=1.3,
            label="Double-minimum case",
        )
        first_time = float(result["first_low_time_h"][station_index])
        first_value = float(result["first_low_nT"][station_index])
        second_time = float(result["second_low_time_h"][station_index])
        second_value = float(result["second_low_nT"][station_index])
        ax.scatter(
            [first_time],
            [first_value],
            marker="o",
            s=21,
            facecolor="white",
            edgecolor=DOUBLE_COLOR,
            linewidth=1.0,
            zorder=7,
        )
        ax.scatter(
            [second_time],
            [second_value],
            marker="s",
            s=18,
            facecolor="white",
            edgecolor=DOUBLE_COLOR,
            linewidth=1.0,
            zorder=7,
        )
        ax.annotate("1", (first_time, first_value), xytext=(3, -9), textcoords="offset points", fontsize=5.4)
        ax.annotate("2", (second_time, second_value), xytext=(3, -9), textcoords="offset points", fontsize=5.4)
        _set_curve_axes(ax, metadata, station_index, panel)
        _metric_box(
            ax,
            "\n".join(
                (
                    rf"1: {format_hhmm(first_time)}, {first_value:.1f} nT",
                    rf"2: {format_hhmm(second_time)}, {second_value:.1f} nT",
                    rf"$\Delta\tau_{{12}}$={result["low_gap_h"][station_index]:.2f} h",
                )
            ),
        )

    _set_shared_limits(axes, [metadata["background_nT"], result["total_nT"]])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.998),
        handlelength=2.5,
    )
    fig.text(
        0.5,
        0.927,
        rf"$A_a={config['anomaly']['A_a_nT']:g}$ nT, "
        rf"$\tau_a={config['time']['tau_a_h']:g}$ h, "
        rf"$p={config['time']['p']:g}$, "
        rf"$z_a={config['anomaly']['z_a_km']:g}$ km",
        ha="center",
        fontsize=6.4,
    )
    fig.supxlabel("Local time", y=0.02)
    fig.supylabel(r"$Z$ (nT)", x=0.015)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.105, top=0.875, wspace=0.22, hspace=0.25)
    save_figure(fig, output_dir, "double_minimum", dpi)
    plt.close(fig)


def write_double_metrics_csv(
    path: Path, metadata: dict[str, Any], result: dict, config: dict
) -> None:
    fields = [
        "station",
        "A_a_nT",
        "tau_a_h",
        "p",
        "z_a_km",
        "background_min_time_h",
        "first_low_time_h",
        "first_low_nT",
        "second_low_time_h",
        "second_low_nT",
        "low_gap_h",
        "total_range_nT",
        "max_abs_anomaly_nT",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, station in enumerate(metadata["station_names"]):
            writer.writerow(
                {
                    "station": station,
                    "A_a_nT": f"{config['anomaly']['A_a_nT']:.8g}",
                    "tau_a_h": f"{config['time']['tau_a_h']:.8g}",
                    "p": f"{config['time']['p']:.8g}",
                    "z_a_km": f"{config['anomaly']['z_a_km']:.8g}",
                    "background_min_time_h": f"{metadata['background_min_time_h'][index]:.8g}",
                    "first_low_time_h": f"{result['first_low_time_h'][index]:.8g}",
                    "first_low_nT": f"{result['first_low_nT'][index]:.10g}",
                    "second_low_time_h": f"{result['second_low_time_h'][index]:.8g}",
                    "second_low_nT": f"{result['second_low_nT'][index]:.10g}",
                    "low_gap_h": f"{result['low_gap_h'][index]:.8g}",
                    "total_range_nT": f"{result['total_range_nT'][index]:.10g}",
                    "max_abs_anomaly_nT": f"{result['max_abs_anomaly_nT'][index]:.10g}",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the six-station double-minimum figure.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    output_dir = args.output_dir.resolve()
    configure_style()
    metadata, result = compute_double_minimum(PARAMETERS)
    plot_double_minimum(metadata, result, PARAMETERS, output_dir, args.dpi)
    write_double_metrics_csv(output_dir / "double_minimum_metrics.csv", metadata, result, PARAMETERS)
    print(f"Saved double-minimum figure and metrics to {output_dir}")


if __name__ == "__main__":
    main()
