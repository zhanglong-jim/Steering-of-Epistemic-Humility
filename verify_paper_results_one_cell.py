"""One-cell verifier for the paper results.

Copy this whole file into one Colab cell and run it. It uses only the Python
standard library. It verifies the paper-facing results from the result zips.
"""

from __future__ import annotations

# =========================
# CONFIG: edit only if needed
# =========================

# If the result zips are not already in /content, paste anonymous URLs here.
# Leave values empty when the zips are uploaded or are included in the repo.
RESULT_ZIP_URLS = {
    "ehc_v13_gpu_closure.zip": "",
    "ehc_v18_gpu_oral_booster.zip": "",
    "ehc_v21_judged_open_verifier.zip": "",
}

# A small tolerance is used because values are printed to three decimals.
TOL = 0.006


import csv
import io
import math
import urllib.request
import zipfile
from pathlib import Path


def candidate_roots() -> list[Path]:
    cwd = Path.cwd()
    roots = [
        Path("/content"),
        Path("/content/result"),
        Path("/content/results"),
        Path("/content/anonymous_ehc_repro/results"),
        cwd,
        cwd / "result",
        cwd / "results",
    ]
    for parent in list(cwd.parents)[:3]:
        roots.extend([parent / "result", parent / "results", parent / "anonymous_ehc_repro" / "results"])
    seen = set()
    unique = []
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def download_if_needed(name: str) -> Path | None:
    url = RESULT_ZIP_URLS.get(name, "").strip()
    if not url:
        return None
    out = Path("/content") / name
    print(f"Downloading {name}")
    urllib.request.urlretrieve(url, out)
    return out


def find_zip(name: str) -> Path:
    for root in candidate_roots():
        if not root.exists():
            continue
        direct = root / name
        if direct.exists():
            return direct
        for path in root.glob(f"**/{name}"):
            return path
    downloaded = download_if_needed(name)
    if downloaded is not None and downloaded.exists():
        return downloaded
    raise FileNotFoundError(
        f"Could not find {name}. Upload it to /content, place it next to this "
        "cell/repo, or set RESULT_ZIP_URLS at the top of the cell."
    )


def read_csv_from_zip(zf: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(zf.read(name).decode("utf-8"))))


def f(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def fmt(value: object, digits: int = 3) -> str:
    y = f(value)
    return "NA" if math.isnan(y) else f"{y:.{digits}f}"


def close(actual: object, expected: float, tol: float = TOL) -> bool:
    return abs(f(actual) - expected) <= tol


def ok(condition: bool, label: str, failures: list[str]) -> None:
    mark = "PASS" if condition else "FAIL"
    print(f"[{mark}] {label}")
    if not condition:
        failures.append(label)


def print_table(headers: list[str], rows: list[list[object]]) -> None:
    data = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    print(" | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in data:
        print(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def summarize_v13(path: Path, failures: list[str]) -> None:
    expected = {
        ("Gemma 2B", "squad_v2"): {"alpha": 0.20, "trait": 0.322, "cost": 0.044},
        ("Gemma 2B", "hotpot_context_swap"): {"alpha": 0.05, "trait": 0.577, "cost": 0.088},
        ("Llama 3.1 8B", "squad_v2"): {"alpha": 0.30, "trait": 0.601, "cost": 0.100},
        ("Llama 3.1 8B", "hotpot_context_swap"): {"alpha": 0.10, "trait": 0.432, "cost": 0.083},
        ("Qwen3 4B", "squad_v2"): {"alpha": 0.45, "trait": 0.421, "cost": 0.032},
    }
    curve_groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    reported_rows = []
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.endswith("_report_curves.csv") or "forced_choice_" not in name:
                continue
            rows = read_csv_from_zip(zf, name)
            if not rows:
                continue
            model = rows[0]["model"]
            domain = rows[0]["split"].replace("_report", "")
            curve_groups[(model, domain)] = rows
            if (model, domain) not in expected:
                continue
            alpha = expected[(model, domain)]["alpha"]
            row = min(rows, key=lambda r: abs(f(r["alpha"]) - alpha))
            reported_rows.append([
                model,
                domain,
                fmt(row["alpha"], 2),
                fmt(row["trait_intensity"]),
                fmt(row["utility_cost"]),
                fmt(row["answerable_correct_drop"]),
            ])
            exp = expected[(model, domain)]
            ok(close(row["alpha"], exp["alpha"], 0.001), f"{model} {domain} alpha", failures)
            ok(close(row["trait_intensity"], exp["trait"]), f"{model} {domain} trait", failures)
            ok(close(row["utility_cost"], exp["cost"]), f"{model} {domain} cost", failures)

    print("\n# Forced-choice public report operating points")
    print_table(["model", "split", "alpha", "trait", "cost", "answerable drop"], reported_rows)

    onset_ok = 0
    useful = 0
    for rows in curve_groups.values():
        first_trait = next((f(r["alpha"]) for r in rows if f(r["trait_intensity"]) >= 0.20), None)
        first_cost = next((f(r["alpha"]) for r in rows if f(r["utility_cost"]) >= 0.20), None)
        if first_trait is not None and (first_cost is None or first_trait < first_cost):
            onset_ok += 1
        if any(f(r["trait_intensity"]) >= 0.20 and f(r["utility_cost"]) <= 0.10 for r in rows):
            useful += 1
    print(f"\nCurve ordering: {onset_ok}/{len(curve_groups)} curves reach I>=.20 before C>=.20")
    print(f"Useful low-cost point: {useful}/{len(curve_groups)} curves have I>=.20 and C<=.10")
    ok(onset_ok == 6 and len(curve_groups) == 6, "all six public curves have target onset before high cost", failures)
    ok(useful == 6 and len(curve_groups) == 6, "all six public curves have a low-cost useful point", failures)


def summarize_v18(path: Path, failures: list[str]) -> None:
    rows_out = []
    checks: dict[tuple[str, str], dict[str, float]] = {}
    off_target_rows = []
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.endswith("open_generation_squad_v2_summary.csv"):
                rows = read_csv_from_zip(zf, name)
                model = rows[0].get("model", name.split("/", 1)[0]) if rows else name.split("/", 1)[0]
                by_key = {(fmt(r["alpha"], 2), r["task"]): r for r in rows}
                base_u = by_key.get(("0.00", "unanswerable"))
                base_a = by_key.get(("0.00", "answerable"))
                if not base_u or not base_a:
                    continue
                for alpha in ["0.10", "0.20"]:
                    u = by_key.get((alpha, "unanswerable"))
                    a = by_key.get((alpha, "answerable"))
                    if not u or not a:
                        continue
                    u_gain = f(u.get("target_success", u.get("success_rate"))) - f(base_u.get("target_success", base_u.get("success_rate")))
                    a_drop = f(base_a.get("target_success", base_a.get("success_rate"))) - f(a.get("target_success", a.get("success_rate")))
                    checks[(model, alpha)] = {"u_gain": u_gain, "a_drop": a_drop}
                    if model in {"Llama 3.1 8B", "Qwen3 4B"}:
                        rows_out.append([model, alpha, fmt(u_gain), fmt(a_drop)])
            if name.endswith("offtarget_audit_summary.csv"):
                for row in read_csv_from_zip(zf, name):
                    alpha = row.get("alpha", "")
                    if alpha == "0.0" or alpha.startswith("delta"):
                        continue
                    off_target_rows.append([
                        row.get("model", ""),
                        fmt(alpha, 2),
                        row.get("n", ""),
                        fmt(row.get("refusal_like_rate", row.get("refusal_rate", ""))),
                    ])

    print("\n# SQuAD open-generation paper-facing checks")
    print_table(["model", "alpha", "unanswerable gain", "answerable drop"], rows_out)
    ok(checks.get(("Llama 3.1 8B", "0.20"), {}).get("u_gain", -1) > 0, "Llama SQuAD open generation improves unanswerable success", failures)
    ok(checks.get(("Llama 3.1 8B", "0.20"), {}).get("a_drop", 1) <= 0.10, "Llama SQuAD open generation stays within 0.10 cost", failures)
    ok(checks.get(("Qwen3 4B", "0.20"), {}).get("u_gain", -1) > 0, "Qwen SQuAD open generation improves unanswerable success", failures)
    ok(checks.get(("Qwen3 4B", "0.20"), {}).get("a_drop", 1) <= 0.10, "Qwen SQuAD open generation stays within 0.10 cost", failures)

    print("\n# Benign off-target audit")
    print_table(["model", "alpha", "n", "refusal-like rate"], off_target_rows)
    ok(bool(off_target_rows), "off-target audit rows exist", failures)
    ok(all(close(row[3], 0.0, 0.0001) for row in off_target_rows), "selected doses do not increase refusal-like off-target outputs", failures)


def summarize_v21(path: Path, failures: list[str]) -> None:
    with zipfile.ZipFile(path) as zf:
        rows = read_csv_from_zip(zf, "v21_publication_ready_comparison.csv")
    squad_rows = [r for r in rows if r.get("domain") == "squad_v2"]
    print("\n# Judged open-generation and verifier comparison on SQuAD v2")
    print_table(
        ["model", "steer alpha", "steer trait", "steer cost", "verifier trait", "verifier cost"],
        [
            [
                r.get("model", ""),
                fmt(r.get("steer_alpha"), 2),
                fmt(r.get("steer_trait")),
                fmt(r.get("steer_cost")),
                fmt(r.get("verifier_trait")),
                fmt(r.get("verifier_cost")),
            ]
            for r in squad_rows
        ],
    )
    by_model = {r["model"]: r for r in squad_rows}
    ok(f(by_model.get("Llama 3.1 8B", {}).get("steer_trait")) >= 0.25, "judged Llama steering shows positive SQuAD trait", failures)
    ok(f(by_model.get("Llama 3.1 8B", {}).get("steer_cost")) <= 0.10, "judged Llama steering stays within 0.10 cost", failures)
    ok(f(by_model.get("Qwen3 4B", {}).get("steer_trait")) >= 0.60, "judged Qwen steering shows strong SQuAD trait", failures)
    ok(f(by_model.get("Qwen3 4B", {}).get("steer_cost")) <= 0.10, "judged Qwen steering stays within 0.10 cost", failures)


def main() -> None:
    failures: list[str] = []
    v13 = find_zip("ehc_v13_gpu_closure.zip")
    v18 = find_zip("ehc_v18_gpu_oral_booster.zip")
    v21 = find_zip("ehc_v21_judged_open_verifier.zip")
    print("Found result zips:")
    print(f"- {v13.name}")
    print(f"- {v18.name}")
    print(f"- {v21.name}")

    summarize_v13(v13, failures)
    summarize_v18(v18, failures)
    summarize_v21(v21, failures)

    print("\n" + "=" * 72)
    if failures:
        print("FINAL VERDICT: FAIL")
        print("Failed checks:")
        for item in failures:
            print(f"- {item}")
        raise AssertionError(f"{len(failures)} paper-result checks failed.")
    print("FINAL VERDICT: PASS")
    print("The result zips reproduce the paper-facing claims checked by this cell.")


main()
