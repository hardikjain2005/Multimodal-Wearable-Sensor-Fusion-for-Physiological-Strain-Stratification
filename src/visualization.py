"""Static report figures (matplotlib/seaborn).

Colour policy: strain classes use the first three slots of a CVD-validated categorical
palette (these three pass all-pairs colour-blind checks), always in the same mapping so a
class keeps its colour in every figure. Sequential data (heatmaps) use one hue light->dark;
correlations use a blue<->red diverging map with a neutral midpoint.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR, STRAIN_ORDER

CLASS_COLORS = {"Low": "#2a78d6", "Moderate": "#1baf7a", "High": "#eb6834"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "seq_blue", ["#f7fafe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "div", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a92", "#c22f2f"])
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False,
})


def save(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png")
    plt.close(fig)


# ------------------------------------------------------------------ data understanding
def plot_sensor_availability(inventory: pd.DataFrame, streams: list[str]):
    """Q: which sensors exist in which session? (shows the modality/label confound)"""
    mat = pd.DataFrame({s: (inventory[f"{s}_rows"] > 0).astype(int) for s in streams})
    mat.index = inventory.session_id
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(mat) + 1.2))
    sns.heatmap(mat, cmap=matplotlib.colors.ListedColormap(["#f0efec", "#2a78d6"]), cbar=False,
                linewidths=2, linecolor="white", ax=ax, annot=mat.replace({1: "✓", 0: ""}), fmt="",
                annot_kws={"color": "white", "fontsize": 9})
    ax.set_xlabel(""); ax.set_ylabel(""); ax.grid(False)
    ax.set_title("Sensor availability per session (blue = recorded)")
    save(fig, "01_sensor_availability")


def plot_session_overview(windows_clean: pd.DataFrame):
    """Q: how much usable data does each class/split have? (class balance, session durations)"""
    g = (windows_clean.groupby(["split", "strain_label", "session_id"]).size()
         .rename("minutes").reset_index())
    g["strain_label"] = pd.Categorical(g.strain_label, STRAIN_ORDER, ordered=True)
    g = g.sort_values(["split", "strain_label", "session_id"])
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(g) + 1.2))
    y = np.arange(len(g))
    ax.barh(y, g.minutes, color=[CLASS_COLORS[c] for c in g.strain_label], height=0.7)
    ax.set_yticks(y, [f"{s}  [{sp.replace('test_', 'test: ')}]" for s, sp in zip(g.session_id, g.split)])
    ax.invert_yaxis(); ax.set_xlabel("Usable 60-s windows (= minutes) after quality control")
    for yi, m in zip(y, g.minutes):
        ax.text(m + 3, yi, str(m), va="center", fontsize=8, color=MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[c]) for c in STRAIN_ORDER]
    ax.legend(handles, STRAIN_ORDER, title="Strain proxy", loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("Usable data per session, coloured by activity-derived strain label")
    ax.grid(axis="y", visible=False)
    save(fig, "02_session_windows")


def plot_class_distribution(windows_clean: pd.DataFrame):
    """Q: is the development set imbalanced?"""
    ct = windows_clean.groupby(["split", "strain_label"]).size().unstack().reindex(columns=STRAIN_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(ct)); w = 0.26
    for i, c in enumerate(STRAIN_ORDER):
        bars = ax.bar(x + (i - 1) * w, ct[c], width=w - 0.02, color=CLASS_COLORS[c], label=c)
        ax.bar_label(bars, fmt="%d", fontsize=8, color=MUTED, padding=2)
    ax.set_xticks(x, ct.index); ax.set_ylabel("Windows (60 s)")
    ax.set_title("Class distribution per data split"); ax.legend(ncols=3, loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, "03_class_distribution")


def plot_feature_distributions(df: pd.DataFrame, features: list[str], name: str, title: str, ncols: int = 3):
    """Q: how does each physiological feature differ between strain levels?"""
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.9 * nrows))
    for ax, f in zip(np.atleast_1d(axes).ravel(), features):
        d = df[["strain_label", f]].dropna()
        sns.boxplot(data=d, x="strain_label", y=f, order=[c for c in STRAIN_ORDER if c in set(d.strain_label)],
                    hue="strain_label", palette=CLASS_COLORS, legend=False, ax=ax, width=0.55,
                    fliersize=1.5, linewidth=0.9)
        ax.set_xlabel(""); ax.set_title(f, fontsize=10)
        ax.grid(axis="x", visible=False)
    for ax in np.atleast_1d(axes).ravel()[len(features):]:
        ax.set_visible(False)
    fig.suptitle(title, fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, name)


def plot_correlation_heatmap(df: pd.DataFrame, features: list[str], name: str = "08_feature_correlation"):
    """Q: which features carry redundant information? (Spearman, robust to outliers)"""
    corr = df[features].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(0.42 * len(features) + 2, 0.42 * len(features) + 1.5))
    sns.heatmap(corr, cmap=DIV_CMAP, vmin=-1, vmax=1, center=0, square=True, linewidths=0.5,
                linecolor="white", cbar_kws={"shrink": 0.6, "label": "Spearman ρ"}, ax=ax)
    ax.set_title("Feature correlation (development set)"); ax.grid(False)
    save(fig, name)
    return corr


# ------------------------------------------------------------------ model evaluation
def plot_confusion_pair(pairs: dict, name: str, title: str):
    """Q: which strain levels does each model confuse? (row-normalised = recall per class)"""
    from sklearn.metrics import confusion_matrix
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.2 * len(pairs), 4.2))
    for ax, (label, (y, pred, _)) in zip(np.atleast_1d(axes), pairs.items()):
        present = [c for c in STRAIN_ORDER if c in set(y)]
        cm = confusion_matrix(y, pred, labels=STRAIN_ORDER)[[STRAIN_ORDER.index(c) for c in present]]
        pct = cm / cm.sum(1, keepdims=True) * 100
        annot = np.array([[f"{p:.0f}%\n({n})" for p, n in zip(pr, nr)] for pr, nr in zip(pct, cm)])
        sns.heatmap(pct, annot=annot, fmt="", cmap=SEQ_CMAP, vmin=0, vmax=100, cbar=False, ax=ax,
                    xticklabels=STRAIN_ORDER, yticklabels=present, linewidths=2, linecolor="white",
                    annot_kws={"fontsize": 10})
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(label, fontsize=10); ax.grid(False)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    save(fig, name)


def plot_roc_pair(pairs: dict, name: str, title: str):
    """Q: how well do predicted probabilities rank each class, independent of the threshold?"""
    from sklearn.metrics import roc_curve, auc
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.2))
    for ax, (label, (y, _, proba)) in zip(np.atleast_1d(axes), pairs.items()):
        for i, c in enumerate(STRAIN_ORDER):
            yb = (np.asarray(y) == c).astype(int)
            if 0 < yb.sum() < len(yb):
                fpr, tpr, _ = roc_curve(yb, proba[:, i])
                ax.plot(fpr, tpr, color=CLASS_COLORS[c], lw=2, label=f"{c} vs rest (AUC {auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
        ax.set_xlabel("False positive rate (1 − specificity)"); ax.set_ylabel("True positive rate (sensitivity)")
        ax.set_title(label, fontsize=10); ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    save(fig, name)


def plot_model_comparison(res: pd.DataFrame, name: str, title: str, metric: str = "f1_macro"):
    """Q: does performance improve with more feature groups, and with the ensemble?"""
    feature_sets = list(dict.fromkeys(res.feature_set))
    models = list(dict.fromkeys(res.model))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.8 / len(models)
    x = np.arange(len(feature_sets))
    for i, m in enumerate(models):
        vals = [res[(res.model == m) & (res.feature_set == f)][metric].mean() for f in feature_sets]
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * w, vals, width=w - 0.02, color=MODEL_COLORS[i], label=m)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, color=MUTED, padding=1.5)
    ax.set_xticks(x, feature_sets); ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(0, 1.08); ax.set_title(title); ax.grid(axis="x", visible=False)
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    save(fig, name)


def plot_metric_bars(df: pd.DataFrame, label_col: str, metrics: list[str], name: str, title: str):
    """Q: how do precision / recall / F1 compare across a few configurations?"""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(metrics)); w = 0.8 / len(df)
    for i, (_, r) in enumerate(df.iterrows()):
        bars = ax.bar(x + (i - (len(df) - 1) / 2) * w, [r[m] for m in metrics], width=w - 0.02,
                      color=MODEL_COLORS[i], label=r[label_col])
        ax.bar_label(bars, fmt="%.2f", fontsize=7, color=MUTED, padding=1.5)
    ax.set_xticks(x, [m.replace("_", " ") for m in metrics]); ax.set_ylim(0, 1.08)
    ax.set_title(title); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=min(3, len(df)), fontsize=8)
    save(fig, name)


# ------------------------------------------------------------------ explainability
def plot_shap_summary(sv, sample: pd.DataFrame, feats: list[str], classes: list[str], name: str):
    """Q: which features push a window towards each strain level? (one beeswarm per class)"""
    import shap
    fig, axes = plt.subplots(1, len(classes), figsize=(5.5 * len(classes), 6))
    for i, (ax, c) in enumerate(zip(axes, classes)):
        plt.sca(ax)
        shap.plots.beeswarm(sv[:, :, i], max_display=12, show=False, plot_size=None, color_bar=(i == len(classes) - 1))
        ax.set_title(f"P({c})", color=CLASS_COLORS.get(c, TEXT)); ax.set_xlabel("SHAP value (change in probability)")
    fig.suptitle("SHAP: feature contributions to each class probability (proposed model, development windows)",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, name)


def plot_shap_groups(grp: pd.DataFrame, name: str):
    """Q: how much does each sensor-derived feature group contribute overall and per class?"""
    share = grp / grp.sum()
    cols = [c for c in STRAIN_ORDER if c in share.columns] + ["overall"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    left = np.zeros(len(cols))
    for i, g in enumerate(share.index):
        vals = share.loc[g, cols].to_numpy()
        ax.barh(cols, vals, left=left, color=MODEL_COLORS[i], label=g, height=0.6, edgecolor="white", linewidth=2)
        for y, (l, v) in enumerate(zip(left, vals)):
            if v > 0.06:
                ax.text(l + v / 2, y, f"{v:.0%}", ha="center", va="center", color="white", fontsize=9)
        left += vals
    ax.set_xlim(0, 1); ax.set_xlabel("Share of mean |SHAP|"); ax.invert_yaxis()
    ax.set_title("Contribution of each feature group (proposed model)")
    ax.legend(ncols=len(share), loc="upper center", bbox_to_anchor=(0.5, -0.2)); ax.grid(False)
    save(fig, name)


# ------------------------------------------------------------------ healthcare insights
def plot_hr_overlap(dev: pd.DataFrame, name: str):
    """Q: why does heart rate alone fail? Yoga (Moderate) and hike (High) overlap in HR,
    but separate along an ECG-derived feature."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
    for c in STRAIN_ORDER:
        d = dev[dev.strain_label == c]
        sns.kdeplot(d.hr_mean, ax=axes[0], color=CLASS_COLORS[c], fill=True, alpha=0.25, lw=2, label=c, cut=0)
        axes[1].scatter(d.hr_mean, d.ecg_hf_noise_ratio, s=9, alpha=0.45, color=CLASS_COLORS[c], label=c,
                        edgecolors="none")
    axes[0].set_xlabel("Mean heart rate in window (bpm)"); axes[0].set_ylabel("Density")
    axes[0].set_title("HR alone: Moderate overlaps Low and High"); axes[0].legend()
    axes[1].set_yscale("log"); axes[1].set_xlabel("Mean heart rate (bpm)")
    axes[1].set_ylabel("ECG high-frequency noise ratio (log)")
    axes[1].set_title("An ECG-derived feature helps separate Moderate")
    axes[1].legend(markerscale=2)
    fig.tight_layout()
    save(fig, name)


def plot_session_profiles(dev: pd.DataFrame, features: dict, name: str):
    """Q: how does each recording's typical physiology differ by strain level?
    One dot = one session's median (the unit used in the statistical tests)."""
    med = dev.groupby(["session_id", "strain_label"])[list(features)].median().reset_index()
    fig, axes = plt.subplots(1, len(features), figsize=(2.6 * len(features), 3.4))
    for ax, (f, label) in zip(axes, features.items()):
        for i, c in enumerate(STRAIN_ORDER):
            v = med.loc[med.strain_label == c, f].dropna()
            ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)) if len(v) > 1 else [i],
                       v, s=60, color=CLASS_COLORS[c], edgecolors="white", linewidths=1.5, zorder=3)
        ax.set_xticks(range(3), STRAIN_ORDER, fontsize=8); ax.set_title(label, fontsize=9)
        ax.set_xlim(-0.5, 2.5); ax.grid(axis="x", visible=False)
    fig.suptitle("Session-level physiological profile (development sessions; one dot = one recording)",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, name)


def plot_test_session_predictions(sess: pd.DataFrame, name: str):
    """Q: on each unseen test recording, what share of minutes does each model assign to each level?"""
    models = list(dict.fromkeys(sess.label))
    sessions = list(dict.fromkeys(sess.session_id))
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 0.5 * len(sessions) + 1.3), sharey=True)
    for ax, m in zip(axes, models):
        d = sess[sess.label == m].set_index("session_id").loc[sessions]
        left = np.zeros(len(d))
        for c in STRAIN_ORDER:
            v = d[f"pred_{c}_pct"].to_numpy()
            ax.barh(range(len(d)), v, left=left, color=CLASS_COLORS[c], height=0.65, edgecolor="white",
                    linewidth=2, label=f"predicted {c}")
            for y, (l, val) in enumerate(zip(left, v)):
                if val >= 12:
                    ax.text(l + val / 2, y, f"{val:.0f}%", ha="center", va="center", color="white", fontsize=8)
            left += v
        ax.set_yticks(range(len(d)), [f"{s}  (true: {t})" for s, t in zip(d.index, d.true_label)])
        ax.set_xlim(0, 100); ax.invert_yaxis(); ax.set_title(m, fontsize=10); ax.grid(False)
        ax.set_xlabel("% of windows")
    axes[0].legend(ncols=3, loc="upper center", bbox_to_anchor=(1.05, -0.18))
    fig.suptitle("Held-out test sessions: share of 60-s windows assigned to each strain level", fontweight="bold")
    save(fig, name)
