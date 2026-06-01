from __future__ import annotations

import json
import math
import os
import random
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit

import reddit_hpv_modern_proposal_pipeline as base


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = base.INPUT_PATH
INPUT_PATH = Path(os.environ.get("REDDIT_HPV_INPUT_PATH", str(DEFAULT_INPUT_PATH))).resolve()
RUN_NAME = os.environ.get("REDDIT_HPV_RUN_NAME", INPUT_PATH.stem)


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


DEFAULT_RULE_REFERENCE_PATH = BASE_DIR / "analysis_outputs" / "reddit_hpv_cervical_analytics_v2" / "concept_prevalence.csv"
rule_reference_env = os.environ.get("REDDIT_HPV_RULE_REFERENCE_PATH")
if rule_reference_env:
    RULE_REFERENCE_PATH: Path | None = Path(rule_reference_env).resolve()
elif INPUT_PATH == DEFAULT_INPUT_PATH:
    RULE_REFERENCE_PATH = DEFAULT_RULE_REFERENCE_PATH
else:
    RULE_REFERENCE_PATH = None

output_dir_env = os.environ.get("REDDIT_HPV_OUT_DIR")
if output_dir_env:
    OUT_DIR = Path(output_dir_env).resolve()
elif INPUT_PATH == DEFAULT_INPUT_PATH:
    OUT_DIR = BASE_DIR / "analysis_outputs" / "reddit_hpv_modern_proposal_pipeline_v2"
else:
    OUT_DIR = BASE_DIR / "analysis_outputs" / f"reddit_hpv_modern_proposal_pipeline_v2__{_slugify(RUN_NAME)}"
TABLES_DIR = OUT_DIR / "tables"
FIGURES_DIR = OUT_DIR / "figures"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42
TRAIN_FRAC = 0.75
MIN_POS_FOR_MODEL = 10
MIN_NEG_FOR_MODEL = 20
MIN_CAL_POS = 5
MIN_CAL_NEG = 5
MAX_NEG_MULTIPLIER = 4
DEFAULT_HIGH_CONF_DELTA = 0.12
DEFAULT_HIGH_CONF_FLOOR = 0.8

random.seed(SEED)
np.random.seed(SEED)
plt.switch_backend("Agg")
sns.set_theme(style="whitegrid", context="talk")

RULE_MAP = {
    "internalized_stigma": "sexual_self_blame_or_judgment",
    "relational_stigma": "partner_disclosure_or_rejection",
    "transmission_anxiety": "transmission_or_reinfection_anxiety",
    "cancer_fear": "cancer_fear_or_spiral",
    "procedure_anxiety": "procedure_pain_fear",
    "confusion_uncertainty": "confusion_or_information_gap",
    "information_seeking": "information_seeking",
    "support_seeking": "peer_or_partner_support",
    "supplement_use": "supplement_use",
    "lifestyle_change": "lifestyle_change",
    "self_advocacy": "provider_switch_or_self_advocacy",
    "cognitive_reframing": "cognitive_reframing",
}

EXTENSION_PATTERNS = {
    "internalized_stigma": [
        r"\bmy fault\b",
        r"\bblame myself\b",
        r"\bi feel guilty\b",
        r"\bi felt guilty\b",
        r"\bi have some shame\b",
        r"\b(?:i feel|i felt|i am|i'm|im|i was|i've felt|ive felt|feel like i(?:'m| am)|like i(?:'m| am)|makes me feel)\b.{0,25}\b(?:ashamed?|embarrass\w*|dirty|gross|tainted|disgusting|broken|unworthy)\b",
        r"\bam i\b.{0,12}\b(?:dirty|gross|tainted)\b",
        r"\bslut\b",
        r"\bpromiscuous\b",
        r"\bregret who i slept with\b",
    ],
    "relational_stigma": [
        r"\bdisclos\w*\b",
        r"\bdating\b",
        r"\bpartner\b",
        r"\breject\w*\b",
        r"\bintimate\b",
        r"\bsex life\b",
        r"\btell (?:him|her|them|my partner)\b",
    ],
    "transmission_anxiety": [
        r"\btransmi\w*\b",
        r"\breinfect\w*\b",
        r"\bgive (?:it )?to\b",
        r"\bpass(?:ed)? (?:it )?to\b",
        r"\bcatch(?:ing)? it\b",
        r"\bcondoms?\b",
    ],
    "cancer_fear": [
        r"\bscared\b",
        r"\bterrified\b",
        r"\bafraid\b",
        r"\banxious\b",
        r"\bpanic\w*\b",
        r"\bspiral\w*\b",
        r"\bworr(?:ied|y)\b",
        r"\bcancer\b",
        r"\bdevastated\b",
    ],
    "procedure_anxiety": [
        r"\bcolposcopy\b",
        r"\bbiopsy\b",
        r"\bleep\b",
        r"\bckc\b",
        r"\bprocedure\b",
        r"\bpain\w*\b",
        r"\bhurt\b",
        r"\banxiety med\b",
        r"\blocal anesth\w*\b",
        r"\bcervical block\b",
    ],
    "confusion_uncertainty": [
        r"\bconfused\b",
        r"\bdoesn'?t make sense\b",
        r"\btrying to make sense\b",
        r"\bwhat now\b",
        r"\bwhat do i do\b",
        r"\bnext steps\b",
        r"\bnot sure\b",
        r"\bunsure\b",
        r"\bdon'?t understand\b",
        r"\bcan someone explain\b",
        r"\blooking for clarification\b",
        r"\bhelp with understanding\b",
        r"\bwhat does (?:this|that) mean\b",
    ],
    "information_seeking": [
        r"\bgoogl\w*\b",
        r"\bresearch\b",
        r"\bsearching\b",
        r"\breading (?:studies|this sub)\b",
        r"\blook(?:ing)? up\b",
        r"\bwiki\b",
        r"\bresources\b",
        r"\bclarification\b",
    ],
    "support_seeking": [
        r"\banyone else\b",
        r"\bhas anyone\b",
        r"\bdid anyone else\b",
        r"\bneed reassurance\b",
        r"\blooking for support\b",
        r"\bplease tell me\b",
        r"\bi feel alone\b",
        r"\bsimilar experience\b",
        r"\bshare your experience\b",
        r"\bany advice\b",
        r"\bcomfort\b",
    ],
    "supplement_use": [
        r"\bahcc\b",
        r"\bzinc\b",
        r"\bfolic acid\b",
        r"\bvitamins?\b",
        r"\bsupplements?\b",
        r"\bturkey tail\b",
        r"\bpapilocare\b",
        r"\bpapillex\b",
        r"\bgreen tea\b",
    ],
    "lifestyle_change": [
        r"\bdiet\b",
        r"\bexercise\b",
        r"\bwalking\b",
        r"\bwalks?\b",
        r"\byoga\b",
        r"\bstress\b",
        r"\bsleep\b",
        r"\bquit smoking\b",
        r"\bstopped smoking\b",
        r"\balcohol\b",
    ],
    "self_advocacy": [
        r"\bsecond opinion\b",
        r"\badvocat\w*\b",
        r"\basked for\b",
        r"\bpushed for\b",
        r"\binsisted on\b",
        r"\brequested\b",
        r"\bswitch(?:ed)? (?:doctor|gyno|obgyn|gynecologist)\b",
        r"\bchange(?:d)? my (?:doctor|gyno|obgyn|gynecologist)\b",
        r"\bpain management\b",
        r"\blocal anesth\w*\b",
    ],
    "cognitive_reframing": [
        r"\bcommon\b",
        r"\bmost people\b",
        r"\bso common\b",
        r"\byou are not alone\b",
        r"\byou will be okay\b",
        r"\bit will be okay\b",
        r"\bgive your body time\b",
        r"\bbody can clear\b",
        r"\btemporary\b",
        r"\btrying not to spiral\b",
    ],
}

HIGH_CONF_FLOOR_BY_LABEL = {
    "confusion_uncertainty": 0.84,
    "support_seeking": 0.84,
    "self_advocacy": 0.84,
    "internalized_stigma": 0.84,
    "transmission_anxiety": 0.84,
    "relational_stigma": 0.82,
    "procedure_anxiety": 0.78,
    "cancer_fear": 0.78,
    "information_seeking": 0.8,
    "supplement_use": 0.8,
    "lifestyle_change": 0.8,
    "cognitive_reframing": 0.8,
}


def ensure_dirs() -> None:
    for path in [OUT_DIR, TABLES_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def compile_patterns() -> dict[str, dict[str, list[re.Pattern[str]]]]:
    compiled: dict[str, dict[str, list[re.Pattern[str]]]] = {}
    for label, spec in base.LABEL_SPECS.items():
        compiled[label] = {"pos": [], "neg": [], "ext": []}
        for patterns in spec["positive_lfs"].values():
            compiled[label]["pos"].extend(re.compile(p, flags=re.IGNORECASE) for p in patterns)
        for patterns in spec["negative_lfs"].values():
            compiled[label]["neg"].extend(re.compile(p, flags=re.IGNORECASE) for p in patterns)
        compiled[label]["ext"].extend(re.compile(p, flags=re.IGNORECASE) for p in EXTENSION_PATTERNS[label])
    return compiled


def build_seed_frame(sentence_df: pd.DataFrame) -> pd.DataFrame:
    compiled = compile_patterns()
    df = sentence_df.copy()

    all_positive_cols: list[str] = []
    for label in base.LABEL_ORDER:
        pos_col = f"seed_pos__{label}"
        neg_col = f"seed_neg__{label}"
        df[pos_col] = df["sentence"].apply(lambda text: any(p.search(text) for p in compiled[label]["pos"]))
        df[neg_col] = df["sentence"].apply(lambda text: any(p.search(text) for p in compiled[label]["neg"]))
        df[f"extension_hit__{label}"] = df["sentence"].apply(lambda text: any(p.search(text) for p in compiled[label]["ext"]))
        all_positive_cols.append(pos_col)

    df["any_seed_positive"] = df[all_positive_cols].sum(axis=1)
    df["neutral_candidate"] = df["any_seed_positive"] == 0

    for label in base.LABEL_ORDER:
        pos_col = f"seed_pos__{label}"
        neg_col = f"seed_neg__{label}"
        other_cols = [c for c in all_positive_cols if c != pos_col]
        other_positive = df[other_cols].sum(axis=1)
        df[f"seed_explicit_positive__{label}"] = df[pos_col] & ~df[neg_col]
        df[f"seed_explicit_negative__{label}"] = df[neg_col] & ~df[pos_col]
        df[f"seed_hard_negative__{label}"] = ~df[pos_col] & ~df[neg_col] & (other_positive > 0)
        df[f"seed_neutral_negative__{label}"] = ~df[pos_col] & ~df[neg_col] & df["neutral_candidate"]

    return df


def embed_sentences(sentences: list[str]) -> np.ndarray:
    model = SentenceTransformer(EMBED_MODEL_NAME, local_files_only=True)
    embeddings = model.encode(
        sentences,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings)


def build_training_index(df: pd.DataFrame, label: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    pos_idx = df.index[df[f"seed_explicit_positive__{label}"]].to_numpy()
    neg_explicit_idx = df.index[df[f"seed_explicit_negative__{label}"]].to_numpy()
    neg_hard_idx = df.index[df[f"seed_hard_negative__{label}"]].to_numpy()
    neg_neutral_idx = df.index[df[f"seed_neutral_negative__{label}"]].to_numpy()

    max_neg = max(MIN_NEG_FOR_MODEL, len(pos_idx) * MAX_NEG_MULTIPLIER)
    neg_pool = list(neg_explicit_idx)

    remaining = max_neg - len(neg_pool)
    if remaining > 0:
        neg_pool.extend(list(np.random.permutation(neg_hard_idx)[:remaining]))
    remaining = max_neg - len(neg_pool)
    if remaining > 0:
        neg_pool.extend(list(np.random.permutation(neg_neutral_idx)[:remaining]))

    neg_idx = np.array(sorted(set(neg_pool)), dtype=int)
    summary = {
        "pos_seed_n": int(len(pos_idx)),
        "neg_explicit_n": int(len(neg_explicit_idx)),
        "neg_hard_available_n": int(len(neg_hard_idx)),
        "neg_neutral_available_n": int(len(neg_neutral_idx)),
        "neg_selected_n": int(len(neg_idx)),
    }
    return pos_idx, neg_idx, summary


def threshold_grid() -> list[float]:
    return [round(x, 2) for x in np.arange(0.35, 0.91, 0.05)]


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray, default_threshold: float) -> tuple[float, float, float, float]:
    if y_true.sum() < MIN_CAL_POS or (len(y_true) - y_true.sum()) < MIN_CAL_NEG:
        return default_threshold, math.nan, math.nan, math.nan

    best_threshold = default_threshold
    best_precision = math.nan
    best_recall = math.nan
    best_f1 = -1.0
    for threshold in threshold_grid():
        y_pred = (y_prob >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )
        if f1 > best_f1:
            best_threshold = float(threshold)
            best_precision = float(precision)
            best_recall = float(recall)
            best_f1 = float(f1)
    return best_threshold, best_precision, best_recall, best_f1


def train_and_score(df: pd.DataFrame, embeddings: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = df.copy()
    summary_rows: list[dict[str, object]] = []

    for label in base.LABEL_ORDER:
        pos_idx, neg_idx, seed_summary = build_training_index(df, label)
        default_threshold = 0.75 if len(pos_idx) < 30 else 0.65

        prob_col = f"model_prob__{label}"
        pred_col = f"model_pred__{label}"
        scored[prob_col] = np.nan
        scored[pred_col] = 0

        if len(pos_idx) < MIN_POS_FOR_MODEL or len(neg_idx) < MIN_NEG_FOR_MODEL:
            scored[prob_col] = scored[f"seed_explicit_positive__{label}"].astype(float)
            scored[pred_col] = scored[f"seed_explicit_positive__{label}"].astype(int)
            summary_rows.append(
                {
                    "label": label,
                    **seed_summary,
                    "train_strategy": "rule_only_fallback",
                    "threshold": 1.0,
                    "cal_precision": math.nan,
                    "cal_recall": math.nan,
                    "cal_f1": math.nan,
                }
            )
            continue

        train_df = pd.concat(
            [
                df.loc[pos_idx, ["thread_id"]].assign(y=1, idx=pos_idx),
                df.loc[neg_idx, ["thread_id"]].assign(y=0, idx=neg_idx),
            ],
            ignore_index=True,
        )
        splitter = GroupShuffleSplit(n_splits=1, train_size=TRAIN_FRAC, random_state=SEED)
        train_i, cal_i = next(splitter.split(train_df, groups=train_df["thread_id"]))
        train_rows = train_df.iloc[train_i]
        cal_rows = train_df.iloc[cal_i]

        model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
        model.fit(embeddings[train_rows["idx"]], train_rows["y"])

        cal_prob = model.predict_proba(embeddings[cal_rows["idx"]])[:, 1]
        threshold, precision, recall, f1 = choose_threshold(
            cal_rows["y"].to_numpy(),
            cal_prob,
            default_threshold=default_threshold,
        )

        full_prob = model.predict_proba(embeddings)[:, 1]
        scored[prob_col] = full_prob

        lexical_seed = scored[f"seed_explicit_positive__{label}"].astype(bool)
        explicit_negative = scored[f"seed_explicit_negative__{label}"].astype(bool)
        extension_hit = scored[f"extension_hit__{label}"].astype(bool)
        high_conf_floor = HIGH_CONF_FLOOR_BY_LABEL.get(label, DEFAULT_HIGH_CONF_FLOOR)
        high_conf_threshold = min(0.95, max(high_conf_floor, threshold + DEFAULT_HIGH_CONF_DELTA))
        final_pred = (
            (
                lexical_seed
                | ((full_prob >= threshold) & extension_hit)
                | (full_prob >= high_conf_threshold)
            )
            & ~explicit_negative
        ).astype(int)
        scored[pred_col] = final_pred

        summary_rows.append(
            {
                "label": label,
                **seed_summary,
                "train_strategy": "embedding_logistic",
                "threshold": threshold,
                "high_conf_threshold": high_conf_threshold,
                "cal_precision": precision,
                "cal_recall": recall,
                "cal_f1": f1,
                "predicted_positive_n": int(final_pred.sum()),
                "mean_probability": round(float(full_prob.mean()), 4),
            }
        )

    return scored, pd.DataFrame(summary_rows)


def aggregate_prevalence(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sentence_rows = []
    source_rows = []
    thread_rows = []
    n_sent = len(scored)
    source_unit_keys = ["thread_id", "source_col"]
    n_source_units = int(scored[source_unit_keys].drop_duplicates().shape[0])
    for label in base.LABEL_ORDER:
        pred_col = f"model_pred__{label}"
        pos_n = int(scored[pred_col].sum())
        sentence_rows.append(
            {
                "label": label,
                "group": base.LABEL_SPECS[label]["group"],
                "sentence_positive_n": pos_n,
                "sentence_positive_pct": round(100 * pos_n / max(n_sent, 1), 2),
                "mean_probability": round(float(scored[f"model_prob__{label}"].mean()), 4),
            }
        )
        source_hits = scored.groupby(source_unit_keys)[pred_col].max()
        source_rows.append(
            {
                "label": label,
                "group": base.LABEL_SPECS[label]["group"],
                "source_positive_n": int(source_hits.sum()),
                "source_positive_pct": round(100 * float(source_hits.mean()), 2),
                "source_unit_n": n_source_units,
            }
        )
        thread_hits = scored.groupby("thread_id")[pred_col].max()
        thread_rows.append(
            {
                "label": label,
                "group": base.LABEL_SPECS[label]["group"],
                "thread_positive_n": int(thread_hits.sum()),
                "thread_positive_pct": round(100 * float(thread_hits.mean()), 2),
            }
        )
    return pd.DataFrame(sentence_rows), pd.DataFrame(source_rows), pd.DataFrame(thread_rows)


def build_rule_comparison(sentence_prev: pd.DataFrame, rule_reference_path: Path | None) -> pd.DataFrame:
    if rule_reference_path is None or not rule_reference_path.exists():
        return pd.DataFrame(columns=["label", "rule_concept", "rule_sentence_pct", "model_sentence_pct", "delta_pct_points"])
    rule_prev = pd.read_csv(rule_reference_path)
    rule_prev = rule_prev[rule_prev["level"] == "concept"][["name", "sentence_count", "sentence_pct", "thread_count"]].copy()
    rows = []
    for label, rule_name in RULE_MAP.items():
        rule_match = rule_prev[rule_prev["name"] == rule_name]
        model_match = sentence_prev[sentence_prev["label"] == label]
        rows.append(
            {
                "label": label,
                "rule_concept": rule_name,
                "rule_sentence_pct": float(rule_match["sentence_pct"].iloc[0]) if not rule_match.empty else np.nan,
                "model_sentence_pct": float(model_match["sentence_positive_pct"].iloc[0]) if not model_match.empty else np.nan,
                "delta_pct_points": (
                    float(model_match["sentence_positive_pct"].iloc[0]) - float(rule_match["sentence_pct"].iloc[0])
                    if (not rule_match.empty and not model_match.empty)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def save_markdown(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_markdown(index=False), encoding="utf-8")


def make_figures(sentence_prev: pd.DataFrame, source_prev: pd.DataFrame, comparison_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_df = sentence_prev.sort_values(["group", "sentence_positive_pct"], ascending=[True, True])
    sns.barplot(data=plot_df, x="sentence_positive_pct", y="label", hue="group", dodge=False, ax=ax)
    ax.set_xlabel("Predicted positive sentences (%)")
    ax.set_ylabel("")
    ax.set_title("V2 Calibrated Reaction Label Prevalence")
    ax.legend(title="Domain", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure1_v2_label_prevalence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8))
    plot_df = source_prev.sort_values(["group", "source_positive_pct"], ascending=[True, True])
    sns.barplot(data=plot_df, x="source_positive_pct", y="label", hue="group", dodge=False, ax=ax)
    ax.set_xlabel("Positive post/comment units (%)")
    ax.set_ylabel("")
    ax.set_title("V2 Post/Comment-Level Reaction Label Prevalence")
    ax.legend(title="Domain", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure2_v2_post_comment_level_prevalence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if not comparison_df.empty and comparison_df["delta_pct_points"].notna().any():
        fig, ax = plt.subplots(figsize=(11, 8))
        plot_df = comparison_df.dropna().sort_values("delta_pct_points")
        sns.barplot(data=plot_df, x="delta_pct_points", y="label", color="#4C78A8", ax=ax)
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("V2 minus earlier rule-based sentence prevalence (percentage points)")
        ax.set_ylabel("")
        ax.set_title("Difference Between V2 and Earlier Rule-Based Prevalence")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "figure3_v2_vs_rule_based_delta.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def build_method_steps(seed_summary: pd.DataFrame, sentence_prev: pd.DataFrame, comparison_available: bool) -> str:
    top_labels = sentence_prev.sort_values("sentence_positive_pct", ascending=False).head(6)
    step_9 = (
        "9. Aggregate predictions back to sentence-level and thread-level prevalence summaries and compare them with the earlier conservative rule-based pipeline."
        if comparison_available
        else "9. Aggregate predictions back to sentence-level and thread-level prevalence summaries; rule-based comparison can be added only when a matched reference run exists for the same corpus."
    )
    method = f"""
**Step-by-Step Methodological Approach**

1. Retrieve Reddit threads and preserve the original post plus associated comment tree per thread.
2. Apply a cervical-screening relevance filter so the analytic corpus is restricted to abnormal Pap / cervical HPV narratives.
3. Segment the corpus into sentence units while retaining thread identifiers for leakage-safe splitting.
4. Define a reaction-oriented label space for stigma, distress, and coping rather than relying on baseline clinical-result labels alone.
5. Construct weak positive seeds from theory-guided lexical cues and construct stronger negative seeds from:
   - explicit anti-patterns within the same label,
   - hard negatives drawn from sentences that match other labels but not the target label,
   - neutral negatives drawn from unlabeled sentences when needed for class balance.
6. Train one classifier per label rather than a single shared multilabel head. This reduces cross-label bleed and makes thresholding label-specific.
7. Represent sentences using local transformer embeddings (`{EMBED_MODEL_NAME}`) and fit balanced logistic classifiers on silver-labeled sentence sets.
8. Tune thresholds on held-out silver calibration sets at the thread level, then apply lexical-semantic gating so a sentence must either:
   - contain an explicit target cue, or
   - exceed the learned threshold by a meaningful confidence margin.
{step_9}
10. Use the resulting annotation workbook and active-learning queue to create a dual-coded gold-standard set for the actual large-scale study.

**What Changed Relative to the Earlier Modern Pipeline**

- Abstentions are now kept as missing rather than as pseudo-0.5 labels.
- Snorkel smoothing is not used when conflict is minimal; direct vote-based seeds are retained.
- Negatives are actively mined instead of assuming unlabeled examples behave as neutral training data.
- Calibration is performed separately for each label.
- Final prediction requires lexical-semantic agreement or very high model confidence.

**Current V2 Pilot Output Pattern**

Top V2 sentence-level labels: {', '.join(f"{row.label} ({row.sentence_positive_pct:.2f}%)" for row in top_labels.itertuples())}.

**Interpretation**

This V2 pipeline is intended to move the pilot toward more realistic prevalence estimates while preserving a modern, scalable architecture. It is still not a replacement for a gold-standard adjudicated training set, but it is methodologically stronger than the earlier degenerate weak-supervision-plus-transformer run because it explicitly addresses sparse coverage, missing negatives, and threshold collapse.
"""
    return textwrap.dedent(method).strip() + "\n"


def main() -> None:
    ensure_dirs()
    raw_df = pd.read_csv(INPUT_PATH)
    thread_df, sentence_df = base.build_sentence_corpus(raw_df)
    seed_df = build_seed_frame(sentence_df)
    embeddings = embed_sentences(seed_df["sentence"].tolist())
    scored_df, seed_summary = train_and_score(seed_df, embeddings)
    sentence_prev, source_prev, thread_prev = aggregate_prevalence(scored_df)
    comparison_df = build_rule_comparison(sentence_prev, RULE_REFERENCE_PATH)
    make_figures(sentence_prev, source_prev, comparison_df)

    seed_df.to_csv(OUT_DIR / "seed_frame.csv", index=False)
    scored_df.to_csv(OUT_DIR / "sentence_predictions_v2.csv", index=False)
    seed_summary.to_csv(OUT_DIR / "seed_and_calibration_summary_v2.csv", index=False)
    sentence_prev.to_csv(OUT_DIR / "label_prevalence_sentence_level_v2.csv", index=False)
    source_prev.to_csv(OUT_DIR / "label_prevalence_post_comment_level_v2.csv", index=False)
    thread_prev.to_csv(OUT_DIR / "label_prevalence_thread_level_v2.csv", index=False)
    if not comparison_df.empty:
        comparison_df.to_csv(OUT_DIR / "comparison_vs_rule_based_v2.csv", index=False)

    save_markdown(seed_summary, TABLES_DIR / "table1_seed_and_calibration_summary_v2.md")
    save_markdown(sentence_prev, TABLES_DIR / "table2_sentence_level_prevalence_v2.md")
    save_markdown(source_prev, TABLES_DIR / "table3_post_comment_level_prevalence_v2.md")
    save_markdown(thread_prev, TABLES_DIR / "table4_thread_level_prevalence_v2.md")
    seed_summary.to_csv(TABLES_DIR / "table1_seed_and_calibration_summary_v2.csv", index=False)
    sentence_prev.to_csv(TABLES_DIR / "table2_sentence_level_prevalence_v2.csv", index=False)
    source_prev.to_csv(TABLES_DIR / "table3_post_comment_level_prevalence_v2.csv", index=False)
    thread_prev.to_csv(TABLES_DIR / "table4_thread_level_prevalence_v2.csv", index=False)
    if not comparison_df.empty:
        save_markdown(comparison_df, TABLES_DIR / "table5_v2_vs_rule_based_comparison.md")
        comparison_df.to_csv(TABLES_DIR / "table5_v2_vs_rule_based_comparison.csv", index=False)
    else:
        (OUT_DIR / "comparison_note.txt").write_text(
            "Rule-based comparison was skipped because no matched rule-based reference run was supplied for this corpus.\n",
            encoding="utf-8",
        )

    methods_text = build_method_steps(seed_summary, sentence_prev, comparison_available=not comparison_df.empty)
    (OUT_DIR / "step_by_step_methodological_approach_v2.md").write_text(methods_text, encoding="utf-8")

    summary = {
        "input_path": str(INPUT_PATH),
        "run_name": RUN_NAME,
        "threads_final_personal_narratives": int(thread_df[(thread_df["include_thread"] == True) & (thread_df["resource_like_thread"] == False)].shape[0]),
        "sentence_units": int(len(sentence_df)),
        "post_comment_units": int(scored_df[["thread_id", "source_col"]].drop_duplicates().shape[0]),
        "embedding_model_name": EMBED_MODEL_NAME,
        "labels_modeled": len(base.LABEL_ORDER),
        "labels_with_embedding_models": int((seed_summary["train_strategy"] == "embedding_logistic").sum()),
        "labels_with_rule_fallback": int((seed_summary["train_strategy"] == "rule_only_fallback").sum()),
        "rule_reference_path": str(RULE_REFERENCE_PATH) if RULE_REFERENCE_PATH is not None else None,
        "rule_comparison_available": bool(not comparison_df.empty),
    }
    (OUT_DIR / "run_summary_v2.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
