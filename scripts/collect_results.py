#!/usr/bin/env python3
"""Collect experiment results and generate LaTeX table rows for all benchmarks.

Usage:
    python3 scripts/collect_results.py [--benchmark all|cifar100|tinyimagenet|imagenet_r]
"""

import json, os, sys, statistics
from pathlib import Path

RUNBOOK = Path("code/runbook")

BENCHMARKS = {
    "cifar100": {
        "dir": "cifar100_main",
        "rank_key": "avgpool_tap",
    },
    "tinyimagenet": {
        "dir": "tinyimagenet_main",
        "rank_key": "avgpool_tap",
    },
    "imagenet_r": {
        "dir": "imagenet_r_main",
        "rank_key": "avgpool_tap",
    },
}

METHODS_ORDER = [
    ("erm", r"\quad ERM"),
    ("ewc", r"\quad EWC"),
    ("agem", r"\quad A-GEM"),
    ("er", r"\quad ER"),
    ("derpp", r"\quad DER++"),
    ("er_ace", r"\quad ER-ACE"),
    ("cls_er", r"\quad CLS-ER"),
    ("xder", r"\quad X-DER"),
    ("pacl", r"\quad \PACL{} (on ERM)"),
    ("pacl_er", r"\quad \PACL{}$+$ER"),
    ("pacl_derpp", r"\quad \PACL{}$+$DER++"),
    ("pacl_er_ace", r"\quad \PACL{}$+$ER-ACE"),
    ("pacl_cls_er", r"\quad \PACL{}$+$CLS-ER"),
    ("pacl_xder", r"\quad \PACL{}$+$X-DER"),
]

PACL_METHODS = {
    "pacl",
    "pacl_er",
    "pacl_derpp",
    "pacl_er_ace",
    "pacl_cls_er",
    "pacl_xder",
}


def load_seeds(bench_dir, method, n_seeds=5):
    results = []
    for seed in range(n_seeds):
        path = RUNBOOK / bench_dir / method / f"seed{seed}" / "summary.json"
        if path.exists():
            results.append(json.load(open(path)))
    return results


def stats(vals):
    if len(vals) == 0:
        return None, None, 0
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s, len(vals)


def fmt(m, s, n):
    if m is None:
        return "---", 0
    return f"${m * 100:.2f} \\pm {s * 100:.2f}$", m * 100


def fmt_rank(m, s, n):
    if m is None:
        return "---", 0
    return f"${m:.2f} \\pm {s:.2f}$", m


def collect_benchmark(bench_name):
    cfg = BENCHMARKS[bench_name]
    bench_dir = cfg["dir"]
    rank_key = cfg["rank_key"]

    all_data = {}
    for method_key, _ in METHODS_ORDER:
        seeds = load_seeds(bench_dir, method_key)
        if not seeds:
            all_data[method_key] = None
            continue

        acc_vals = [s["ACC"] for s in seeds]
        bwt_vals = [s["BWT"] for s in seeds]
        rank_vals = [s["rank_last"][rank_key] for s in seeds]

        acc_m, acc_s, acc_n = stats(acc_vals)
        bwt_m, bwt_s, bwt_n = stats(bwt_vals)
        rank_m, rank_s, rank_n = stats(rank_vals)

        all_data[method_key] = {
            "acc": (acc_m, acc_s, acc_n),
            "bwt": (bwt_m, bwt_s, bwt_n),
            "rank": (rank_m, rank_s, rank_n),
        }

    return all_data


def find_best(all_data, metric):
    """Return (best_key, second_key) for a given metric.
    For ACC and rank: higher is better.
    For BWT: higher (less negative) is better.
    """
    vals = {}
    for k, v in all_data.items():
        if v is None:
            continue
        m = v[metric][0]
        if m is not None:
            vals[k] = m

    if not vals:
        return None, None

    sorted_keys = sorted(vals, key=lambda k: vals[k], reverse=True)
    best = sorted_keys[0] if len(sorted_keys) >= 1 else None
    second = sorted_keys[1] if len(sorted_keys) >= 2 else None
    return best, second


def generate_latex(bench_name):
    all_data = collect_benchmark(bench_name)
    cfg = BENCHMARKS[bench_name]

    acc_best, acc_second = find_best(all_data, "acc")
    bwt_best, bwt_second = find_best(all_data, "bwt")
    rank_best, rank_second = find_best(all_data, "rank")

    print(f"\n{'=' * 60}")
    print(f"Benchmark: {bench_name} ({cfg['dir']})")
    print(f"{'=' * 60}\n")

    n_total = 0
    n_complete = 0

    for method_key, latex_name in METHODS_ORDER:
        n_total += 1
        data = all_data.get(method_key)
        if data is None:
            print(f"  {latex_name:40s} NOT STARTED")
            continue

        n_seeds = data["acc"][2]
        n_complete += 1

        acc_str, acc_val = fmt(*data["acc"])
        bwt_str, bwt_val = fmt(*data["bwt"])
        rank_str, rank_val = fmt_rank(*data["rank"])

        # Bold best, underline second
        if method_key == acc_best:
            acc_str = f"\\textbf{{{acc_str}}}"
        elif method_key == acc_second:
            acc_str = f"\\underline{{{acc_str}}}"

        if method_key == bwt_best:
            bwt_str = f"\\textbf{{{bwt_str}}}"
        elif method_key == bwt_second:
            bwt_str = f"\\underline{{{bwt_str}}}"

        if method_key == rank_best:
            rank_str = f"\\textbf{{{rank_str}}}"
        elif method_key == rank_second:
            rank_str = f"\\underline{{{rank_str}}}"

        # Determine section
        if method_key in PACL_METHODS:
            if method_key == "pacl":
                section = "alone"
            else:
                section = "stacked"
        else:
            section = "baseline"

        print(f"  [{section:8s}] {method_key:12s} seeds={n_seeds}")
        print(f"    ACC: {data['acc'][0] * 100:.2f} ± {data['acc'][1] * 100:.2f}%")
        print(f"    BWT: {data['bwt'][0] * 100:.2f} ± {data['bwt'][1] * 100:.2f}%")
        print(f"    rank: {data['rank'][0]:.2f} ± {data['rank'][1]:.2f}")

    print(f"\n  Complete: {n_complete}/{n_total} methods")

    # Generate LaTeX rows
    print(f"\n--- LaTeX rows ---\n")
    current_section = None
    for method_key, latex_name in METHODS_ORDER:
        data = all_data.get(method_key)
        if data is None:
            # Determine section for placeholder
            if method_key in PACL_METHODS:
                section = "alone" if method_key == "pacl" else "stacked"
            else:
                section = "baseline"

            if section != current_section:
                if section == "baseline":
                    print(f"  \\textit{{Baselines}}            & & & \\\\")
                elif section == "alone":
                    print(f"  \\midrule")
                    print(f"  \\textit{{Our method (alone)}}   & & & \\\\")
                elif section == "stacked":
                    print(f"  \\midrule")
                    print(f"  \\textit{{Our method (stacked)}} & & & \\\\")
                current_section = section

            print(f"  {latex_name:40s} & --- & --- & --- \\\\")
            continue

        n_seeds = data["acc"][2]
        if n_seeds < 5:
            print(f"  % WARNING: {method_key} has only {n_seeds} seeds")

        acc_str, _ = fmt(*data["acc"])
        bwt_str, _ = fmt(*data["bwt"])
        rank_str, _ = fmt_rank(*data["rank"])

        if method_key == acc_best:
            acc_str = f"\\textbf{{{acc_str}}}"
        elif method_key == acc_second:
            acc_str = f"\\underline{{{acc_str}}}"

        if method_key == bwt_best:
            bwt_str = f"\\textbf{{{bwt_str}}}"
        elif method_key == bwt_second:
            bwt_str = f"\\underline{{{bwt_str}}}"

        if method_key == rank_best:
            rank_str = f"\\textbf{{{rank_str}}}"
        elif method_key == rank_second:
            rank_str = f"\\underline{{{rank_str}}}"

        if method_key in PACL_METHODS:
            section = "alone" if method_key == "pacl" else "stacked"
        else:
            section = "baseline"

        if section != current_section:
            if section == "baseline":
                print(f"  \\textit{{Baselines}}            & & & \\\\")
            elif section == "alone":
                print(f"  \\midrule")
                print(f"  \\textit{{Our method (alone)}}   & & & \\\\")
            elif section == "stacked":
                print(f"  \\midrule")
                print(f"  \\textit{{Our method (stacked)}} & & & \\\\")
            current_section = section

        print(
            f"  {latex_name:40s} & {acc_str:28s} & {bwt_str:28s} & {rank_str:28s} \\\\"
        )

    print()


if __name__ == "__main__":
    bench_arg = (
        sys.argv[sys.argv.index("--benchmark") + 1]
        if "--benchmark" in sys.argv
        else "all"
    )

    if bench_arg == "all":
        for b in BENCHMARKS:
            generate_latex(b)
    else:
        generate_latex(bench_arg)
