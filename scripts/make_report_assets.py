"""Generate the figures and LaTeX tables used by docs/report/report.tex.

  python scripts/make_report_assets.py   -> docs/report/figs/*.pdf, docs/report/tables/*.tex

Reads only committed results (results/logs, results/nas, results/benchmark.csv), so the
report can be rebuilt on any machine without the datasets or a GPU.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import RESULTS_DIR, ROOT, STUDENTS  # noqa: E402

OUT = ROOT / "docs" / "report"
FIGS, TABLES = OUT / "figs", OUT / "tables"
NAMES = {"mbv2": "MobileNetV2", "squeezenet": "SqueezeNet", "efflite0": "EfficientNet-Lite0",
         "resnet50": "ResNet-50 (teacher)", "nas": "NAS best"}
# Palette without blue: red / charcoal / green / amber / brown.
RED, GREY, GREEN, AMBER, BROWN = "#B03A2E", "#4D4D4D", "#2E7D32", "#C98A0B", "#6D4C41"

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8.5,
                     "axes.labelsize": 8, "legend.fontsize": 7, "lines.linewidth": 1.3,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.prop_cycle": matplotlib.cycler(color=[RED, GREY, GREEN, AMBER, BROWN])})


def log(run):
    return pd.read_csv(RESULTS_DIR / "logs" / f"{run}.csv")


def fig_q1_curves():
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9), constrained_layout=True)
    t = log("teacher_resnet50")
    axes[0].plot(t.epoch, t.test_acc, color=BROWN)
    axes[0].set_title(f"ResNet-50 teacher ({t.test_acc.max():.2f}%)")
    for ax, s in zip(axes[1:], STUDENTS):
        b, k = log(f"baseline_{s}"), log(f"kd_{s}")
        ax.plot(b.epoch, b.test_acc, color=GREY, ls="--", label=f"baseline ({b.test_acc.max():.2f})")
        ax.plot(k.epoch, k.test_acc, color=RED, label=f"KD ({k.test_acc.max():.2f})")
        ax.set_title(NAMES[s])
        ax.legend(loc="lower right", frameon=False)
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("CIFAR-100 test top-1 (%)")
    fig.savefig(FIGS / "q1_curves.pdf")
    plt.close(fig)


def fig_nas():
    h = pd.read_csv(RESULTS_DIR / "nas" / "history.csv")
    best = json.loads((RESULTS_DIR / "nas" / "best_arch.json").read_text())
    c = best["constraints"]
    fig, (a, b, r) = plt.subplots(1, 3, figsize=(7.0, 2.1), constrained_layout=True)
    style = {"accepted": (GREEN, "o", "accepted"), "pruned": (AMBER, "^", "median-pruned"),
             "rejected": (RED, "x", "rejected (constraint)")}
    for st, (col, m, lab) in style.items():
        d = h[h.status == st]
        a.scatter(d.latency_ms, d.size_mb, c=col, marker=m, s=16, label=lab, lw=1)
    a.axvline(c["max_latency_ms"], color=GREY, ls=":", lw=1)
    a.axhline(c["max_size_mb"], color=GREY, ls=":", lw=1)
    bh = h[h.trial == best["trial"]].iloc[0]
    a.scatter([bh.latency_ms], [bh.size_mb], marker="*", s=90, c="black", zorder=5, label="selected")
    a.set_yscale("log")
    a.set_xlabel("CPU latency, batch 1, 1 thread (ms)")
    a.set_ylabel("fp32 size (MB)")
    a.set_title("Hardware profile of all 40 trials")
    a.legend(frameon=False, loc="lower right", fontsize=6, handletextpad=0.2)

    ok = h[h.status.isin(["accepted", "pruned"])]
    for st, (col, m, lab) in style.items():
        if st == "rejected":
            continue
        d = ok[ok.status == st]
        b.scatter(d.trial, d.val_acc, c=col, marker=m, s=16, lw=1)
    acc = h.val_acc.where(h.status == "accepted").cummax().ffill()
    b.step(h.trial, acc, where="post", color=GREY, lw=1, label="best so far")
    b.axvline(7.5, color=GREY, ls=":", lw=0.8)
    b.set_xlabel("trial")
    b.set_ylabel("proxy val acc (%)")
    b.set_title("Search progress (CIFAR-10 proxy)")
    b.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.08))

    n = log("nas_best")
    r.plot(n.epoch, n.test_acc, color=RED, label="test")
    r.plot(n.epoch, n.train_acc, color=GREY, ls="--", label="train")
    r.set_xlabel("epoch")
    r.set_ylabel("top-1 (%)")
    r.set_title(f"Retrain from scratch ({n.test_acc.max():.2f}%)")
    r.legend(frameon=False, loc="lower right")
    for ax in (a, b, r):
        ax.grid(alpha=0.25, lw=0.5)
    fig.savefig(FIGS / "nas.pdf")
    plt.close(fig)


def fmt(v, nd=2):
    return "--" if pd.isna(v) else f"{v:.{nd}f}"


def table_benchmark():
    path = RESULTS_DIR / "benchmark.csv"
    if not path.exists():
        (TABLES / "benchmark.tex").write_text("\\multicolumn{10}{c}{\\emph{benchmark.csv not generated yet}}\\\\\n")
        return
    df = pd.read_csv(path)
    lines = []
    for r in df.itertuples():
        name = NAMES.get(r.arch, r.arch) + (f" ({r.mode.upper() if r.mode == 'kd' else r.mode})" if isinstance(r.mode, str) and r.mode else "")
        lines.append(" & ".join([
            name, r.dataset.upper().replace("CIFAR", "C-"), f"{r.params / 1e6:.2f}", fmt(r.mflops, 0),
            fmt(r.size_mb), fmt(getattr(r, "onnx_size_mb", float("nan"))),
            fmt(r.torch_latency_ms), fmt(getattr(r, "ort_latency_ms", float("nan"))),
            fmt(r.peak_mem_mb, 1), fmt(r.test_top1)]) + r" \\")
    (TABLES / "benchmark.tex").write_text("\n".join(lines) + "\n")


def table_nas_history():
    h = pd.read_csv(RESULTS_DIR / "nas" / "history.csv")
    top = h[h.status == "accepted"].sort_values("val_acc", ascending=False).head(5)
    lines = [f"{int(r.trial)} & {r.val_acc:.2f} & {r.params / 1e3:.0f}k & {r.size_mb:.3f} & "
             f"{r.latency_ms:.2f} & {r.peak_mem_mb:.1f} & \\texttt{{\\scriptsize {r.desc}}} \\\\"
             for r in top.itertuples()]
    (TABLES / "nas_top.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    fig_q1_curves()
    fig_nas()
    table_benchmark()
    table_nas_history()
    print(f"wrote {sorted(p.name for p in FIGS.iterdir())} and {sorted(p.name for p in TABLES.iterdir())}")
