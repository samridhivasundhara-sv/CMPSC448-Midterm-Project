"""Make every figure in figures/ from the JSON files in results/."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES, LABELS, RESULTS, load_data

# categorical slots (fixed order); families always get the same colour
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
FAM_COLOR = dict(zip(LABELS, C))
MODEL_COLOR = {"tfidf": "#8f8d86", "cnn": C[0], "lstm": C[1]}
MODEL_NAME = {"tfidf": "TF-IDF + LogReg", "cnn": "CNN", "lstm": "BiLSTM"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 150,
    "savefig.bbox": "tight", "legend.frameon": False,
})


def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


def grouped_bars(ax, groups, series, values, errors, colors, labels, ylabel):
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(groups))
    for i, s in enumerate(series):
        v = [values[(g, s)] for g in groups]
        e = [errors.get((g, s), 0) for g in groups]
        bars = ax.bar(x + (i - (n - 1) / 2) * w, v, w * 0.92, yerr=e, color=colors[s],
                      label=labels[s], error_kw={"ecolor": MUTED, "lw": 1, "capsize": 2})
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.015, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_xticks(x, groups)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", ncol=n)


def fig_rq1_rq2(r):
    views = ["input", "output", "both"]
    nice = {"input": "Input only\n(prompt)", "output": "Output only\n(response)",
            "both": "Input + Output"}
    kinds = ["tfidf", "cnn", "lstm"]
    vals = {(nice[v], k): r[f"{k}/{v}"]["f1_mean"] for v in views for k in kinds}
    errs = {(nice[v], k): r[f"{k}/{v}"]["f1_std"] for v in views for k in kinds}
    fig, ax = plt.subplots(figsize=(7, 3.8))
    grouped_bars(ax, [nice[v] for v in views], kinds, vals, errs, MODEL_COLOR, MODEL_NAME,
                 "Test macro-F1")
    ax.axhline(0.2, color=MUTED, ls="--", lw=1, label="chance (0.20)")
    ax.legend(loc="upper right", ncol=4, fontsize=8.5)
    ax.set_title("RQ1/RQ2 — which text identifies the LLM? (5 families, mean ± std of 3 seeds)",
                 fontsize=10, loc="left")
    fig.savefig(FIGURES / "rq1_rq2_views.png")


def fig_confusions(r):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
    for ax, k in zip(axes, ["cnn", "lstm"]):
        cm = np.sum([np.array(run["test"]["confusion"]) for run in r[f"{k}/output"]["runs"]], axis=0)
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.55 else INK)
        ax.set_xticks(range(5), LABELS, rotation=30)
        ax.set_yticks(range(5), LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.grid(False)
        ax.set_title(f"{MODEL_NAME[k]} (output only), row-normalised", fontsize=10)
    fig.savefig(FIGURES / "rq1_confusion.png")


def fig_per_class(r):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    kinds = ["tfidf", "cnn", "lstm"]
    vals, errs = {}, {}
    for k in kinds:
        per = np.array([[run["test"]["per_class_f1"][l] for l in LABELS]
                        for run in r[f"{k}/output"]["runs"]])
        for i, l in enumerate(LABELS):
            vals[(l, k)], errs[(l, k)] = per[:, i].mean(), per[:, i].std()
    grouped_bars(ax, LABELS, kinds, vals, errs, MODEL_COLOR, MODEL_NAME, "Test F1")
    ax.set_title("RQ1 — per-family F1 (output only)", fontsize=10, loc="left")
    fig.savefig(FIGURES / "rq1_per_class.png")


def fig_curves(r):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for k in ["cnn", "lstm"]:
        for i, run in enumerate(r[f"{k}/output"]["runs"]):
            h = pd.DataFrame(run["history"])
            kw = dict(color=MODEL_COLOR[k], lw=2, alpha=1 if i == 0 else 0.35,
                      label=MODEL_NAME[k] if i == 0 else None)
            axes[0].plot(h.epoch, h.train_loss, **kw)
            axes[1].plot(h.epoch, h.val_macro_f1, **kw)
    axes[0].set(xlabel="Epoch", ylabel="Training loss", title="Training loss")
    axes[1].set(xlabel="Epoch", ylabel="Validation macro-F1", title="Validation macro-F1")
    axes[1].legend()
    fig.suptitle("Training curves, output-only (3 seeds; faded = seeds 1–2)", fontsize=10, x=0.02,
                 ha="left")
    fig.savefig(FIGURES / "training_curves.png")


def fig_rq3(r3, r12):
    settings = {"A_to_writing": ("writing", "Train code+math+QA → test writing"),
                "B_qa_to_code": ("code", "Train QA → test code")}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    cols = {"in": "#8f8d86", "cross": C[1], "seen": C[0]}
    for ax, (key, (tgt, title)) in zip(axes, settings.items()):
        kinds = ["tfidf", "cnn", "lstm"]
        x = np.arange(len(kinds))
        bars = {
            "in": ([r3[f"{key}/{k}"]["in_domain_macro_f1_mean"] for k in kinds],
                   "In-domain test (seen task types)"),
            "seen": ([np.mean([run[f"test_{tgt}"]["macro_f1"] for run in r12[f"{k}/output"]["runs"]])
                      for k in kinds], f"{tgt} test, model trained on ALL tasks"),
            "cross": ([r3[f"{key}/{k}"]["cross_domain_macro_f1_mean"] for k in kinds],
                      f"{tgt} test, {tgt} never seen in training"),
        }
        for i, (c, (v, lab)) in enumerate(bars.items()):
            b = ax.bar(x + (i - 1) * 0.26, v, 0.24, color=cols[c], label=lab)
            for bb, val in zip(b, v):
                ax.text(bb.get_x() + bb.get_width() / 2, val + 0.01, f"{val:.2f}",
                        ha="center", fontsize=7.5, color=MUTED)
        ax.set_xticks(x, [MODEL_NAME[k] for k in kinds])
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7.5, loc="upper left")
    axes[0].set_ylabel("Test macro-F1")
    fig.suptitle("RQ3 — cross-task generalisation", fontsize=10, x=0.02, ha="left")
    fig.savefig(FIGURES / "rq3_cross_task.png")


def fig_rq3_per_family(r3):
    """Per-family F1 drop (in-domain minus cross-domain), averaged over the 3 models."""
    settings = {"A_to_writing": "code+math+QA → writing", "B_qa_to_code": "QA → code"}
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(LABELS))
    for i, (key, lab) in enumerate(settings.items()):
        drops = []
        for l in LABELS:
            d = [np.mean([run["in_domain"]["per_class_f1"][l] - run["cross_domain"]["per_class_f1"][l]
                          for run in r3[f"{key}/{k}"]["runs"]]) for k in ("tfidf", "cnn", "lstm")]
            drops.append(np.mean(d))
        b = ax.bar(x + (i - 0.5) * 0.38, drops, 0.36, color=[C[0], C[1]][i], label=lab)
        for bb, v in zip(b, drops):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 0.003, f"{v:.2f}", ha="center",
                    fontsize=7.5, color=MUTED)
    ax.set_xticks(x, LABELS)
    ax.set_ylabel("F1 drop on unseen task\n(in-domain − cross-domain)")
    ax.set_title("RQ3 — which families keep their fingerprint on an unseen task? "
                 "(mean of TF-IDF, CNN, BiLSTM)", fontsize=10, loc="left")
    ax.legend()
    fig.savefig(FIGURES / "rq3_per_family.png")


def fig_rq4_features():
    f = pd.read_csv(RESULTS / "rq4_features.csv")
    show = [("n_words", "Response length (words)", 800), ("n_bullets", "Bullet lines", 15),
            ("n_bold", "Bold spans (**…**)", 15), ("has_colon_intro", "First line ends with ':'", None),
            ("starts_affirmation", "Starts with 'Sure/Certainly/…'", None),
            ("ends_with_offer", "Ends with 'let me know/hope this helps'", None)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
    for ax, (col, title, cap) in zip(axes.flat, show):
        if cap is None:  # binary feature -> share of responses
            v = [f[f.family == l][col].mean() for l in LABELS]
            ax.bar(LABELS, v, color=[FAM_COLOR[l] for l in LABELS], width=0.7)
            for i, val in enumerate(v):
                ax.text(i, val + 0.005, f"{val:.0%}", ha="center", fontsize=8, color=MUTED)
            ax.set_ylabel("Share of responses")
        else:
            data = [f[f.family == l][col].clip(upper=cap) for l in LABELS]
            bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6,
                            medianprops={"color": INK})
            for patch, l in zip(bp["boxes"], LABELS):
                patch.set_facecolor(FAM_COLOR[l])
                patch.set_edgecolor(MUTED)
            ax.set_xticks(range(1, 6), LABELS)
            means = [f[f.family == l][col].mean() for l in LABELS]
            ax.plot(range(1, 6), means, "D", color=INK, ms=4, label="mean")
            ax.legend(fontsize=7, loc="upper right")
        ax.set_title(title, fontsize=9.5)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("RQ4 — stylistic fingerprints per family (boxes = middle 50%, ◆ = mean)",
                 fontsize=10, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "rq4_features.png")


def fig_rq4_importance(r4):
    imp = r4["feature_classifier"]["permutation_importance"]
    top = list(imp.items())[:15][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh([k for k, _ in top], [v for _, v in top], color=C[0], height=0.65)
    ax.set_xlabel("Drop in macro-F1 when the feature is shuffled")
    ax.set_title("RQ4 — permutation importance, style-feature-only classifier", fontsize=10,
                 loc="left")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIGURES / "rq4_importance.png")

    g = r4["feature_classifier"]["group_only"]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    items = list(g.items()) + [("all 29 features", r4["feature_classifier"]["macro_f1"])]
    ax.barh([k for k, _ in items][::-1], [v for _, v in items][::-1], color=C[0], height=0.6)
    for i, (_, v) in enumerate(items[::-1]):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8, color=MUTED)
    ax.axvline(0.2, color=MUTED, ls="--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Test macro-F1 (gradient boosting on that feature group only)")
    ax.set_title("RQ4 — which kind of style feature carries the signal?", fontsize=10, loc="left")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIGURES / "rq4_feature_groups.png")


def fig_rq4_ablation(r4):
    ab = r4["ablation"]
    names = ["original", "no_formatting", "words_only", "first_50_tokens", "content_masked",
             "shuffled_order"]
    nice = {"original": "Original", "no_formatting": "No markdown\n/ layout",
            "words_only": "Words only\n(no punct/case)", "first_50_tokens": "First 50\ntokens",
            "content_masked": "Content words\nmasked", "shuffled_order": "Word order\nshuffled"}
    vals = {(nice[n], k): ab[f"{n}/{k}"]["f1_mean"] for n in names for k in ("cnn", "lstm")}
    errs = {(nice[n], k): ab[f"{n}/{k}"]["f1_std"] for n in names for k in ("cnn", "lstm")}
    fig, ax = plt.subplots(figsize=(9, 3.8))
    grouped_bars(ax, [nice[n] for n in names], ["cnn", "lstm"], vals, errs, MODEL_COLOR,
                 MODEL_NAME, "Test macro-F1")
    ax.set_ylim(0, 0.9)
    ax.set_title("RQ4 — ablations: retrain after removing one kind of signal", fontsize=10,
                 loc="left")
    fig.savefig(FIGURES / "rq4_ablation.png")


def fig_dataset():
    df = load_data()
    ct = pd.crosstab(df.LLM_name, df.task)[["qa", "writing", "code", "math"]]
    fig, ax = plt.subplots(figsize=(6.5, 3))
    left = np.zeros(len(ct))
    tcol = {"qa": C[0], "writing": C[1], "code": C[2], "math": C[3]}
    for t in ct.columns:
        ax.barh(ct.index, ct[t], left=left, color=tcol[t], label=t, edgecolor="white", lw=1.5)
        left += ct[t].values
    ax.set_xlabel("Examples")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    ax.grid(axis="y", visible=False)
    fig.savefig(FIGURES / "dataset_tasks.png")
    m = df.groupby(["LLM_name", "model"]).size().rename("n").reset_index()
    m.to_csv(RESULTS / "dataset_models.csv", index=False)


if __name__ == "__main__":
    fig_dataset()
    r12 = load("rq1_rq2_tfidf_cnn_lstm.json")
    fig_rq1_rq2(r12)
    fig_confusions(r12)
    fig_per_class(r12)
    fig_curves(r12)
    if (RESULTS / "rq3.json").exists():
        fig_rq3(load("rq3.json"), r12)
        fig_rq3_per_family(load("rq3.json"))
    fig_rq4_features()
    r4 = {}
    for p in ("rq4_part123.json", "rq4_part4.json"):
        if (RESULTS / p).exists():
            r4.update(load(p))
    if "feature_classifier" in r4:
        fig_rq4_importance(r4)
    if "ablation" in r4:
        fig_rq4_ablation(r4)
    print("figures written to", FIGURES)
