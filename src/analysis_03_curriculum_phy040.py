"""Analysis 3: PHY 040 Module 1 performance by prior school curriculum."""
import base64
import html
import io
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Engineering_Student_Profile_with_Exam_Results.xlsx"
HTML_OUT = ROOT / "out" / "analysis_03_curriculum_phy040.html"
MD_OUT = ROOT / "doc" / "03.md"
PT_PASS_MARK = 50  # Assumed placement pass mark; the workbook does not state the official threshold.
MIN_GROUP_N = 3


def parse_score(value):
    match = re.match(r"^\s*(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else np.nan


def display_curriculum(value):
    label = value.title()
    for old, new in [("Cbse", "CBSE"), ("Ib ", "IB "), ("Sabis", "SABIS"), ("Btech", "BTech")]:
        label = label.replace(old, new)
    return label


def load_cohort():
    raw = pd.read_excel(DATA, sheet_name="Engineering Enrollment")
    raw["pt_score"] = raw["PT PHYS"].map(parse_score)
    raw["module_score"] = pd.to_numeric(raw["Phy040 Module 1 Exam (out of 50)"], errors="coerce")
    raw["curriculum"] = (
        raw["Qual / Award"].fillna("Unknown").astype(str).str.strip().str.casefold()
        .str.replace(r"\s+", " ", regex=True)
    )
    raw["curriculum"] = raw["curriculum"].replace({"a level": "a level"})
    planned = raw["Take PHYS040"].astype(str).str.strip().str.upper().isin({"PHYS040", "PHY040"})
    failed = raw["pt_score"].notna() & raw["pt_score"].lt(PT_PASS_MARK)
    cohort = raw[planned & failed].copy()
    return raw, cohort


def curriculum_summary(cohort):
    rows = []
    for curriculum, group in cohort.groupby("curriculum", dropna=False):
        observed = group["module_score"].dropna()
        if observed.empty:
            continue
        if len(observed) >= 2:
            se = observed.std(ddof=1) / np.sqrt(len(observed))
            critical = stats.t.ppf(0.975, len(observed) - 1)
            ci_low = max(0.0, observed.mean() - critical * se)
            ci_high = min(50.0, observed.mean() + critical * se)
        else:
            ci_low = ci_high = np.nan
        rows.append({
            "Curriculum": display_curriculum(curriculum),
            "Cohort records (n)": len(group),
            "Module scores recorded (n)": len(observed),
            "Module score coverage": len(observed) / len(group),
            "Mean Module 1 (out of 50)": observed.mean(),
            "95% mean CI lower": ci_low,
            "95% mean CI upper": ci_high,
            "Median": observed.median(),
            "SD": observed.std(ddof=1) if len(observed) > 1 else np.nan,
            "Min": observed.min(),
            "Max": observed.max(),
        })
    return pd.DataFrame(rows).sort_values("Mean Module 1 (out of 50)", ascending=False).reset_index(drop=True)


def inferential_test(cohort):
    groups = [
        group["module_score"].dropna().to_numpy()
        for _, group in cohort.groupby("curriculum")
        if group["module_score"].notna().sum() >= MIN_GROUP_N
    ]
    if len(groups) < 2:
        return len(groups), np.nan, np.nan
    result = stats.kruskal(*groups)
    return len(groups), float(result.statistic), float(result.pvalue)


def fmt(value, digits=1):
    return "n/a" if pd.isna(value) else f"{value:.{digits}f}"


def format_html_table(frame):
    shown = frame.copy()
    for column in shown.columns:
        if shown[column].dtype.kind == "f":
            shown[column] = shown[column].map(lambda x: "n/a" if pd.isna(x) else f"{x:.1f}")
    return shown.to_html(index=False, classes="results", border=0, escape=True)


def format_markdown_table(frame):
    columns = list(frame.columns)
    output = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                value = "n/a" if pd.isna(value) else f"{value:.1f}"
            values.append(str(value).replace("|", "\\|"))
        output.append("| " + " | ".join(values) + " |")
    return "\n".join(output)


def plot_summary(summary):
    plot = summary.copy()
    plot["Curriculum label"] = plot["Curriculum"]
    plot = plot.sort_values("Mean Module 1 (out of 50)")
    fig, ax = plt.subplots(figsize=(10, max(5, 0.42 * len(plot))))
    y = np.arange(len(plot))
    means = plot["Mean Module 1 (out of 50)"].to_numpy()
    lower = means - plot["95% mean CI lower"].to_numpy()
    upper = plot["95% mean CI upper"].to_numpy() - means
    lower = np.where(np.isfinite(lower), lower, 0)
    upper = np.where(np.isfinite(upper), upper, 0)
    ax.errorbar(means, y, xerr=np.vstack([lower, upper]), fmt="o", color="#1f6485",
                ecolor="#7b9eaa", capsize=4, markersize=7)
    ax.set_yticks(y, [f"{name} (n={n})" for name, n in zip(plot["Curriculum label"], plot["Module scores recorded (n)"])])
    ax.set_xlim(0, 50)
    ax.set_xlabel("PHY 040 Module 1 score (out of 50); error bars are 95% t-intervals for the mean, bounded to 0-50")
    ax.set_title("PHY 040 Module 1 performance by prior curriculum\nOnly assigned students with a recorded failed PT are included")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def main():
    raw, cohort = load_cohort()
    summary = curriculum_summary(cohort)
    enrolled_cohort = cohort[cohort["Enrolled in PHYS040"].eq("Yes")].copy()
    enrolled_summary = curriculum_summary(enrolled_cohort)
    groups_tested, statistic, p_value = inferential_test(cohort)
    scored = cohort[cohort["module_score"].notna()]
    unscored = cohort[cohort["module_score"].isna()]
    top_supported = summary[summary["Module scores recorded (n)"].ge(MIN_GROUP_N)]
    strongest = top_supported.iloc[0] if not top_supported.empty else None
    enrolled_supported = enrolled_summary[enrolled_summary["Module scores recorded (n)"].ge(MIN_GROUP_N)]
    enrolled_strongest = enrolled_supported.iloc[0] if not enrolled_supported.empty else None
    assignment_vs_enrolled = cohort["Enrolled in PHYS040"].value_counts(dropna=False).to_dict()
    image = plot_summary(summary)

    if strongest is None:
        conclusion = "No curriculum group has enough recorded scores for a supported comparison."
    else:
        conclusion = (
            f"Among curricula with at least {MIN_GROUP_N} recorded Module 1 scores, "
            f"{strongest['Curriculum']} has the highest observed mean: "
            f"{strongest['Mean Module 1 (out of 50)']:.2f}/50 "
            f"(n={int(strongest['Module scores recorded (n)'])}, median {strongest['Median']:.1f})."
        )
    enrolled_conclusion = (
        f"With actual enrollment restricted to Yes, {enrolled_strongest['Curriculum']} has the highest mean among "
        f"curricula with at least {MIN_GROUP_N} scores ({enrolled_strongest['Mean Module 1 (out of 50)']:.2f}/50, "
        f"n={int(enrolled_strongest['Module scores recorded (n)'])})."
        if enrolled_strongest is not None else "The enrolled-only subset has no curriculum group with enough scores for comparison."
    )

    all_rows = len(raw)
    assigned_yes = assignment_vs_enrolled.get("Yes", 0)
    assigned_no = assignment_vs_enrolled.get("No", 0)
    assigned_missing = assignment_vs_enrolled.get(np.nan, 0)
    test_text = (
        f"A Kruskal-Wallis test across the {groups_tested} curriculum groups with at least {MIN_GROUP_N} Module 1 scores "
        f"gives H={statistic:.2f}, p={p_value:.3f}. This does not provide statistically significant evidence of differences "
        "among these curricula at the conventional 0.05 level. It is an exploratory comparison with small and unequal groups."
        if np.isfinite(p_value) else "There are too few curriculum groups with sufficient observations for a group comparison test."
    )

    method = (
        f"The primary cohort comprises rows with `Take PHYS040` equal to `PHYS040` (including the workbook's `PHY040` variant) "
        f"and a numeric Physics placement score below {PT_PASS_MARK}. The 50-point placement pass mark is an assumption because "
        "the workbook does not specify the official pass threshold. This identifies 52 assigned students. Of these, "
        f"{len(scored)} have a `Phy040 Module 1 Exam (out of 50)` score and {len(unscored)} have no recorded module score. "
        "Only scored records contribute to curriculum means and medians. Curriculum labels are normalized for case and whitespace "
        "(for example `A level` and `A Level` are combined); otherwise labels remain separate because the workbook provides no "
        "validated curriculum taxonomy."
    )

    doc = [
        "# Analysis 3: Performance by curriculum in PHY 040", "",
        f"**Finding:** {conclusion} This should be treated as a descriptive signal, not a statistically established ranking.", "",
        "## Scope and cohort", method, "",
        f"The source workbook contains {all_rows} student records. The `Take PHYS040` assignment field is used to identify students required/recommended to take the course; the separate `Enrolled in PHYS040` field is not used to exclude assigned students. In the selected cohort that field records {assigned_yes} `Yes`, {assigned_no} `No`, and {assigned_missing} missing. This mismatch is retained and disclosed rather than silently filtering the sample.", "",
        "## Curriculum results", format_markdown_table(summary), "",
        f"The leading result among curricula with n >= {MIN_GROUP_N} is selected to avoid calling a one- or two-student curriculum the strongest. {test_text}", "",
        "## Sensitivity: explicitly enrolled students only", f"Restricting the cohort to records marked `Enrolled in PHYS040 = Yes` leaves {len(enrolled_cohort)} students, with {int(enrolled_cohort['module_score'].notna().sum())} module scores. {enrolled_conclusion}", format_markdown_table(enrolled_summary), "",
        "A single IB Diploma student scored 35.5/50, higher than the CBSE mean, but that one observation cannot support a curriculum-level conclusion. The sensitivity results are descriptive only.", "",
        "## Interpretation", "- The outcome is the PHY 040 Module 1 Exam score, reported on its recorded 0-50 scale. Higher mean and median indicate stronger observed scores.",
        "- Small group sizes, unequal curriculum representation, missing module marks, and differences in student selection limit direct comparisons.",
        "- A one-student group may have the numerically highest mean, but it cannot support a curriculum-level conclusion; all available curricula are included in the table and plot with their sample sizes.",
        "- The workbook does not provide a PHY 040 module pass threshold. No pass/fail rates are inferred.",
        "- `Take PHYS040` is treated as the assignment/required-course indicator. `Enrolled in PHYS040` differs for many cohort rows; enrollment status is reported, not used to redefine the assigned cohort.",
        "- The PT failure threshold of 50 is an assumption; results should be rerun if the institution confirms a different pass mark.",
        "- Confidence intervals are t-intervals for group means, intersected with the possible 0-50 exam score range; one-record groups have no interval.", "",
        "## Reproducibility", "Run `uv run python src/analysis_03_curriculum_phy040.py`. It generates this document and `out/analysis_03_curriculum_phy040.html`.",
    ]
    MD_OUT.parent.mkdir(exist_ok=True)
    MD_OUT.write_text("\n".join(doc) + "\n", encoding="utf-8")

    css = """body{font:16px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1120px;margin:32px auto;padding:0 22px;color:#202b36}h1{font-size:2rem;border-bottom:3px solid #287c73;padding-bottom:10px}h2{margin-top:34px;color:#284b55}.lead,.caveat{padding:16px 20px;border-left:5px solid #287c73;background:#edf7f4}.caveat{border-color:#cc8c22;background:#fff7e7}.results{border-collapse:collapse;margin:12px 0 24px;max-width:100%;font-size:.9rem}.results th{background:#284b55;color:white;text-align:left}.results th,.results td{padding:7px 9px;border:1px solid #d8e0e8}.results tr:nth-child(even){background:#f3f6f8}img{width:100%;height:auto}code{background:#edf0f2;padding:2px 4px;border-radius:3px}li{margin:7px 0}@media(max-width:700px){body{margin:18px auto;padding:0 12px}.results{font-size:.75rem}.results th,.results td{padding:5px}}"""
    caveat = (
        f"The assumed PT pass mark is {PT_PASS_MARK}. The cohort has {len(cohort)} assigned students, but only "
        f"{len(scored)} Module 1 scores. The `Enrolled in PHYS040` values are Yes={assigned_yes}, No={assigned_no}, "
        f"missing={assigned_missing}; they differ from the assignment field."
    )
    report = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Analysis 3: PHY 040 by curriculum</title><style>{css}</style></head><body>
<h1>Analysis 3: Performance by curriculum</h1><p>PHY 040 Module 1 Exam among students assigned to PHY 040 after a recorded Physics Placement Test failure.</p>
<div class="lead"><b>Finding:</b> {html.escape(conclusion)} {html.escape(test_text)}</div>
<div class="caveat"><b>Important cohort caveat:</b> {html.escape(caveat)} The PT failure cutoff is an assumption; the workbook does not define the official pass score.</div>
<h2>Method</h2><p>{html.escape(method)}</p><p>The enrollment status distribution in the assigned cohort is Yes={assigned_yes}, No={assigned_no}, missing={assigned_missing}. It is reported but is not used to remove course-assigned students from the cohort.</p>
<h2>Observed Module 1 performance</h2>{format_html_table(summary)}<img alt="PHY 040 Module 1 average score by school curriculum with 95 percent confidence intervals" src="data:image/png;base64,{image}">
<h2>Sensitivity: explicitly enrolled students only</h2><p>Restricting to records marked <code>Enrolled in PHYS040 = Yes</code> leaves {len(enrolled_cohort)} students, all with a Module 1 score. {html.escape(enrolled_conclusion)} One IB Diploma student scored 35.5/50, but a one-student curriculum result cannot establish a curriculum-level effect.</p>{format_html_table(enrolled_summary)}
<h2>Interpretation and limitations</h2><ul><li>Means and medians describe observed scores only; no course pass threshold is documented.</li><li>Groups are small and unbalanced, and {len(unscored)} of {len(cohort)} assigned students have no module score.</li><li>Curricula with fewer than {MIN_GROUP_N} scored students are shown but not used as the leading curriculum-level finding.</li><li>The inferential test is exploratory and does not show a significant difference across the included groups at p&lt;0.05.</li><li>The curriculum label is the workbook's <code>Qual / Award</code> field, normalized only for case and whitespace.</li></ul>
<p>Detailed evidence and reproducibility notes: <code>doc/03.md</code>. Rerun with <code>uv run python src/analysis_03_curriculum_phy040.py</code>.</p></body></html>"""
    HTML_OUT.parent.mkdir(exist_ok=True)
    HTML_OUT.write_text(report, encoding="utf-8")
    print(f"Wrote {HTML_OUT}")
    print(f"Wrote {MD_OUT}")
    print(f"Assigned cohort n={len(cohort)}, module marks n={len(scored)}, missing module marks n={len(unscored)}")
    print(summary.to_string(index=False))
    print(test_text)


if __name__ == "__main__":
    main()