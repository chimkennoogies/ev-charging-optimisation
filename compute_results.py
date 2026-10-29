"""
Compute model comparison results using battery-health-centric metrics.

Primary metric: battery_degradation (lower is better)
Secondary metrics: battery-health style metrics provided in input (e.g. retention, efficiency, thermal stress)

Usage:
    python3 compute_results.py --input outputs/model_comparison_metrics.json --our-model "Our Model"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


REQUIRED_COLUMNS = {"model", "battery_degradation"}
LOWER_IS_BETTER_HINTS = ("degradation", "stress", "temperature", "resistance", "fade", "loss")
HIGHER_IS_BETTER_HINTS = ("efficiency", "retention", "health", "soh", "lifetime", "life")
ERROR_METRIC_TERMS = ("rmse", "mae", "mse", "mape", "r2", "median_ae", "error")


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {col: col.strip().lower().replace(" ", "_") for col in df.columns}
    df = df.rename(columns=renamed)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    df["battery_degradation"] = pd.to_numeric(df["battery_degradation"], errors="coerce")
    if df["battery_degradation"].isna().any():
        raise ValueError("All 'battery_degradation' values must be numeric.")

    if "optimized" in df.columns:
        df["optimized"] = df["optimized"].astype(str).str.lower().isin({"true", "1", "yes"})
    else:
        df["optimized"] = True

    for col in df.columns:
        if col in {"model", "optimized"}:
            continue
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


def _is_error_metric(name: str) -> bool:
    return any(term in name for term in ERROR_METRIC_TERMS)


def _infer_metric_direction(metric_name: str) -> str:
    if any(hint in metric_name for hint in HIGHER_IS_BETTER_HINTS):
        return "higher_is_better"
    return "lower_is_better"


def _find_health_metrics(df: pd.DataFrame) -> List[Tuple[str, str]]:
    metrics: List[Tuple[str, str]] = []
    for col in df.columns:
        if col in {"model", "optimized"}:
            continue
        if _is_error_metric(col):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        metrics.append((col, _infer_metric_direction(col)))

    if "battery_degradation" not in [name for name, _ in metrics]:
        raise ValueError("Missing required numeric metric: battery_degradation")
    return metrics


def _resolve_existing_path(input_path: Path) -> Path:
    """Resolve file path from cwd or script directory, with .js -> .json typo handling."""
    script_dir = Path(__file__).resolve().parent

    if not input_path.is_absolute() and not input_path.exists():
        candidate = script_dir / input_path
        if candidate.exists():
            input_path = candidate

    if not input_path.exists():
        if input_path.suffix.lower() == ".js":
            corrected = input_path.with_suffix(".json")
            if corrected.exists():
                input_path = corrected
            else:
                corrected_in_script_dir = script_dir / corrected
                if corrected_in_script_dir.exists():
                    input_path = corrected_in_script_dir
                else:
                    raise FileNotFoundError(
                        f"Input file not found: {input_path}. "
                        f"Did you mean: {corrected}?"
                    )
        else:
            if not input_path.is_absolute():
                candidate = script_dir / input_path
                if candidate.exists():
                    input_path = candidate
                else:
                    parent = input_path.parent if str(input_path.parent) else Path(".")
                    script_parent = candidate.parent
                    candidates = sorted(parent.glob("model_comparison_metrics.*"))
                    script_candidates = sorted(script_parent.glob("model_comparison_metrics.*"))
                    combined = candidates + [p for p in script_candidates if p not in candidates]
                    if combined:
                        suggestions = ", ".join(str(p) for p in combined[:3])
                        raise FileNotFoundError(
                            f"Input file not found: {input_path}. "
                            f"Available similar files: {suggestions}"
                        )
                    raise FileNotFoundError(f"Input file not found: {input_path}")
            else:
                raise FileNotFoundError(f"Input file not found: {input_path}")
    return input_path


def _load_records_from_file(input_path: Path) -> List[Dict]:
    input_path = _resolve_existing_path(input_path)

    if input_path.suffix.lower() == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        records = payload["models"] if isinstance(payload, dict) and "models" in payload else payload
        if isinstance(records, dict):
            records = [records]
        if not isinstance(records, list):
            raise ValueError(f"Unsupported JSON shape in {input_path}. Expected list or {{'models': [...]}}.")
        return records
    elif input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path).to_dict("records")
    else:
        raise ValueError(f"Unsupported input file type for {input_path}. Use .json or .csv")


def _load_input(input_path: Path) -> pd.DataFrame:
    records = _load_records_from_file(input_path)
    return _normalize_dataframe(pd.DataFrame(records))


def _load_model_inputs(input_paths: List[str]) -> pd.DataFrame:
    all_records: List[Dict] = []
    for raw_path in input_paths:
        all_records.extend(_load_records_from_file(Path(raw_path)))
    if not all_records:
        raise ValueError("No model records found in --model-inputs files.")
    return _normalize_dataframe(pd.DataFrame(all_records))


def _create_no_optimization_baseline_from_sessions(session_data_path: Path) -> Dict:
    session_data_path = _resolve_existing_path(session_data_path)
    df = pd.read_csv(session_data_path)

    required = [
        "State of Charge (Start %)",
        "State of Charge (End %)",
        "Temperature (°C)",
        "Charger Type",
        "Charging Rate (kW)",
        "Battery Capacity (kWh)",
        "Energy Consumed (kWh)",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Cannot build baseline from session data. Missing required column(s): "
            f"{missing}"
        )

    soc_start = pd.to_numeric(df["State of Charge (Start %)"], errors="coerce").fillna(50).clip(0, 100)
    soc_end = pd.to_numeric(df["State of Charge (End %)"], errors="coerce").fillna(80).clip(0, 100)
    temp = pd.to_numeric(df["Temperature (°C)"], errors="coerce").fillna(25)
    charge_rate = pd.to_numeric(df["Charging Rate (kW)"], errors="coerce").fillna(7).clip(lower=0)
    battery_cap = pd.to_numeric(df["Battery Capacity (kWh)"], errors="coerce").fillna(60).clip(lower=1e-3)
    energy = pd.to_numeric(df["Energy Consumed (kWh)"], errors="coerce").fillna(0).clip(lower=0)

    fast_charge = df["Charger Type"].astype(str).str.lower().str.contains("dc").astype(float)
    soc_gain = (soc_end - soc_start).clip(lower=0)
    high_soc = ((soc_end - 80) / 20).clip(lower=0, upper=1)
    temp_stress = ((temp - 25).abs() / 20).clip(lower=0, upper=1)
    c_rate = (charge_rate / battery_cap).clip(lower=0, upper=2) / 2

    # No-optimization behavior assumptions: more high-SOC charging and higher fast-charge usage.
    fast_unoptimized = (0.3 + 0.7 * fast_charge).clip(0, 1)
    high_soc_unoptimized = (0.4 + 0.6 * high_soc).clip(0, 1)
    temp_unoptimized = (0.2 + 0.8 * temp_stress).clip(0, 1)

    degradation_index = (
        0.45 * fast_unoptimized
        + 0.25 * high_soc_unoptimized
        + 0.20 * temp_unoptimized
        + 0.10 * c_rate
    )
    battery_degradation = float((0.02 + 0.03 * degradation_index.mean()))
    thermal_stress_index = float(temp_unoptimized.mean())

    soc_gain_kwh = (soc_gain / 100.0) * battery_cap
    efficiency = (soc_gain_kwh / energy.replace(0, pd.NA)).astype(float).fillna(0.9).clip(0, 1)
    charging_efficiency = float((efficiency * 0.92).mean())  # penalty for no optimization

    capacity_retention_pct = float(max(0.0, 100.0 - battery_degradation * 250.0))

    return {
        "model": "No Optimization Baseline",
        "optimized": False,
        "battery_degradation": battery_degradation,
        "capacity_retention_pct": capacity_retention_pct,
        "charging_efficiency": charging_efficiency,
        "thermal_stress_index": thermal_stress_index,
    }


def _infer_baseline(df: pd.DataFrame, baseline_name: str | None) -> str:
    if baseline_name:
        if baseline_name not in set(df["model"]):
            raise ValueError(f"Baseline model '{baseline_name}' not found in input.")
        return baseline_name

    non_optimized = df.loc[~df["optimized"], "model"].tolist()
    if non_optimized:
        return non_optimized[0]

    keywords = ("no optimization", "no_optimization", "baseline")
    for model_name in df["model"].tolist():
        if any(k in str(model_name).lower() for k in keywords):
            return model_name

    raise ValueError(
        "Could not infer no-optimization baseline. "
        "Pass --baseline with the baseline model name."
    )


def _build_health_comparison(df: pd.DataFrame, our_model: str, baseline: str) -> Dict:
    if our_model not in set(df["model"]):
        raise ValueError(f"Our model '{our_model}' not found in input.")

    health_metrics = _find_health_metrics(df)

    ranked = df.sort_values("battery_degradation", ascending=True).reset_index(drop=True).copy()
    ranked["rank_battery_degradation"] = ranked.index + 1

    our_val = float(df.loc[df["model"] == our_model, "battery_degradation"].iloc[0])
    baseline_val = float(df.loc[df["model"] == baseline, "battery_degradation"].iloc[0])

    absolute_reduction = baseline_val - our_val
    pct_reduction = (absolute_reduction / baseline_val * 100.0) if baseline_val != 0 else 0.0

    vs_others: List[Dict] = []
    for _, row in df.iterrows():
        name = str(row["model"])
        if name == our_model:
            continue
        other = float(row["battery_degradation"])
        vs_others.append(
            {
                "model": name,
                "battery_degradation": other,
                "absolute_reduction_vs_model": other - our_val,
                "percent_reduction_vs_model": ((other - our_val) / other * 100.0) if other != 0 else 0.0,
            }
        )

    secondary = {}
    for metric_name, direction in health_metrics:
        if metric_name == "battery_degradation":
            continue
        our_metric = float(df.loc[df["model"] == our_model, metric_name].iloc[0])
        baseline_metric = float(df.loc[df["model"] == baseline, metric_name].iloc[0])
        if direction == "lower_is_better":
            absolute_delta = baseline_metric - our_metric
            percent_delta = (absolute_delta / baseline_metric * 100.0) if baseline_metric != 0 else 0.0
        else:
            absolute_delta = our_metric - baseline_metric
            percent_delta = (absolute_delta / baseline_metric * 100.0) if baseline_metric != 0 else 0.0
        secondary[metric_name] = {
            "direction": direction,
            "our_model_value": our_metric,
            "baseline_value": baseline_metric,
            "absolute_improvement_vs_baseline": absolute_delta,
            "percent_improvement_vs_baseline": percent_delta,
        }

    return {
        "our_model": our_model,
        "baseline_model": baseline,
        "error_metrics_removed": True,
        "battery_degradation": {
            "our_model_value": our_val,
            "baseline_value": baseline_val,
            "absolute_reduction_vs_baseline": absolute_reduction,
            "percent_reduction_vs_baseline": pct_reduction,
        },
        "secondary_health_metrics": secondary,
        "ranking": ranked[["model", "battery_degradation", "rank_battery_degradation"]].to_dict("records"),
        "vs_other_models": sorted(vs_others, key=lambda x: x["battery_degradation"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare models using battery degradation and similar health metrics.")
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/model_comparison_metrics.json",
        help="Path to one JSON/CSV file containing model metrics.",
    )
    parser.add_argument(
        "--model-inputs",
        nargs="+",
        default=None,
        help="One or more JSON/CSV files for different optimization models.",
    )
    parser.add_argument("--our-model", type=str, default="Our Model", help="Name of your model in input data.")
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Name of the no-optimization baseline model. If omitted, script will infer.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/battery_degradation_comparison.json",
        help="Path to save computed comparison results.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="outputs/battery_degradation_ranking.csv",
        help="Path to save battery degradation ranking table.",
    )
    parser.add_argument(
        "--create-baseline-from-data",
        action="store_true",
        help="Auto-create a no-optimization baseline from session data and add/replace it in comparison.",
    )
    parser.add_argument(
        "--session-data",
        type=str,
        default="data/processed/processed_data.csv",
        help="Session dataset used to build no-optimization baseline.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)

    if not output_json.is_absolute():
        output_json = script_dir / output_json
    if not output_csv.is_absolute():
        output_csv = script_dir / output_csv

    if args.model_inputs:
        df = _load_model_inputs(args.model_inputs)
    else:
        df = _load_input(Path(args.input))

    if args.create_baseline_from_data:
        baseline_record = _create_no_optimization_baseline_from_sessions(Path(args.session_data))
        df = df.loc[df["model"] != baseline_record["model"]].copy()
        df = pd.concat([df, pd.DataFrame([baseline_record])], ignore_index=True)

    baseline = _infer_baseline(df, args.baseline)
    report = _build_health_comparison(df, args.our_model, baseline)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    pd.DataFrame(report["ranking"]).to_csv(output_csv, index=False)

    summary = report["battery_degradation"]
    print("\nBattery Degradation Comparison")
    print(f"Our model: {report['our_model']}")
    print(f"No-optimization baseline: {report['baseline_model']}")
    print(f"Our model degradation: {summary['our_model_value']:.6f}")
    print(f"Baseline degradation: {summary['baseline_value']:.6f}")
    print(f"Absolute reduction vs baseline: {summary['absolute_reduction_vs_baseline']:.6f}")
    print(f"Percent reduction vs baseline: {summary['percent_reduction_vs_baseline']:.2f}%")
    if args.create_baseline_from_data:
        print("Baseline source: created from session data (no-optimization assumptions)")
    if report["secondary_health_metrics"]:
        print("\nSecondary battery-health metrics:")
        for metric_name, values in report["secondary_health_metrics"].items():
            print(
                f"  {metric_name}: our={values['our_model_value']:.6f}, "
                f"baseline={values['baseline_value']:.6f}, "
                f"improvement={values['percent_improvement_vs_baseline']:.2f}%"
            )
    print(f"\nSaved JSON report to: {output_json}")
    print(f"Saved ranking CSV to: {output_csv}")


if __name__ == "__main__":
    main()
