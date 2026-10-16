"""Analysis 2: school-mark thresholds for CBSE and Pakistani BISE students."""
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
HTML_OUT = ROOT / "out" / "analysis_02_cbse_bise.html"
MD_OUT = ROOT / "doc" / "02.md"
THRESHOLDS = [75, 70, 60]
PLACEMENT_PASS_MARK = 50

SUBJECTS = {
    "Mathematics": {
        "school": "Math Mark", "placement": "PT MATH",
        "module": "Math 042 Module one results out of 100",
        "other": "Math 141 Midterm Exam (out of 100)", "course": "MATH 042",
    },
    "Physics": {
        "school": "Phys Mark", "placement": "PT PHYS",
        "module": "Phy040 Module 1 Exam (out of 50)",
        "other": None, "course": "PHY 040",
    },
}


def numeric(value):
    match = re.match(r"^\s*(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else np.nan


def load_data():
    raw = pd.read_excel(DATA, sheet_name="Engineering Enrollment")
    qualification = raw["Qual / Award"].astype(str).str.strip().str.casefold()
    cbse = raw[qualification.eq("cbse")].copy()
    pakistan = raw[raw["Nationality"].astype(str).str.contains("pakistan", case=False, na=False)]
    for subject, columns in SUBJECTS.items():
        columns["school_num"] = f"{subject}_school_pct"
        columns["placement_num"] = f"{subject}_placement_score"
        columns["module_num"] = f"{subject}_module_score"
        cbse[columns["school_num"]] = cbse[columns["school"]].map(numeric)
        cbse[columns["placement_num"]] = cbse[columns["placement"]].map(numeric)
        cbse[columns["module_num"]] = pd.to_numeric(cbse[columns["module"]], errors="coerce")
        if columns["other"]:
            columns["other_num"] = f"{subject}_other_score"
            cbse[columns["other_num"]] = pd.to_numeric(cbse[columns["other"]], errors="coerce")
    return raw, cbse, pakistan


def threshold_summary(df, subject, threshold):
    cfg = SUBJECTS[subject]
    school = df[cfg["school_num"]]
    placement = df[cfg["placement_num"]]
    module = df[cfg["module_num"]]
    eligible = school.ge(threshold)
    tested = eligible & placement.notna()
    below_tested = school.lt(threshold) & placement.notna()
    scores = placement[tested]
    modules = module[eligible & module.notna()]
    successes = int((scores >= PLACEMENT_PASS_MARK).sum())
    ci = stats.binomtest(successes, len(scores)).proportion_ci(method="wilson") if len(scores) else None
    return {
        "Subject": subject,
        "School mark cutoff": f"{threshold}%",
        "Eligible with known mark (n)": int(eligible.sum()),
        "Placement scores recorded (n)": int(tested.sum()),
        "Placement coverage": tested.sum() / eligible.sum() if eligible.sum() else np.nan,
        "PT pass rate among tested": (scores >= PLACEMENT_PASS_MARK).mean() if len(scores) else np.nan,
        "PT pass Wilson 95% CI": f"{ci.low:.0%}-{ci.high:.0%}" if ci else "not estimable",
        "Tested below cutoff (n)": int(below_tested.sum()),
        "PT pass rate below cutoff": (placement[below_tested] >= PLACEMENT_PASS_MARK).mean() if below_tested.any() else np.nan,
        "Later module scores recorded (n)": int(len(modules)),
        "Mean later module score": modules.mean() if len(modules) else np.nan,
        "Median later module score": modules.median() if len(modules) else np.nan,
    }


def all_summaries(df):
    return pd.DataFrame([
        threshold_summary(df, subject, threshold)
        for subject in SUBJECTS
        for threshold in THRESHOLDS
    ])


def html_table(frame, percent_columns=()):
    result = frame.copy()
    for column in result.columns:
        if column in percent_columns:
            result[column] = result[column].map(lambda x: "n/a" if pd.isna(x) else f"{x:.0%}")
        elif result[column].dtype.kind == "f":
            result[column] = result[column].map(lambda x: "n/a" if pd.isna(x) else f"{x:.1f}")
    return result.to_html(index=False, classes="data-table", border=0, escape=True)


def markdown_table(frame, percent_columns=()):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column in percent_columns:
                value = "n/a" if pd.isna(value) else f"{value:.0%}"
            elif isinstance(value, (float, np.floating)):
                value = "n/a" if pd.isna(value) else f"{value:.1f}"
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def plot_data(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    palette = {75: "#b2182b", 70: "#ef8a62", 60: "#2166ac"}
    for ax, (subject, cfg) in zip(axes, SUBJECTS.items()):
        school, placement = df[cfg["school_num"]], df[cfg["placement_num"]]
        valid = school.notna() & placement.notna()
        ax.scatter(school[valid], placement[valid], alpha=0.78, edgecolor="white", linewidth=0.6, color="#385f8a")
        for threshold in THRESHOLDS:
            ax.axvline(threshold, color=palette[threshold], linestyle="--", label=f"School cutoff {threshold}%")
        ax.axhline(PLACEMENT_PASS_MARK, color="#333333", linestyle=":", label=f"Assumed PT pass {PLACEMENT_PASS_MARK}")
        ax.set(xlim=(0, 105), ylim=(0, 105), xlabel=f"Recorded {subject.lower()} school mark (%)",
               ylabel="Placement test score", title=f"{subject}: school mark vs placement score\n(n={int(valid.sum())} with both recorded)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def later_summary(df):
    rows = []
    for subject, cfg in SUBJECTS.items():
        measures = [(cfg["module"], cfg["module_num"])]
        if cfg["other"]:
            measures.append((cfg["other"], cfg["other_num"]))
        for label, score_col in measures:
            scores = pd.to_numeric(df[score_col], errors="coerce")
            for threshold in THRESHOLDS:
                selected = df[cfg["school_num"]].ge(threshold) & scores.notna()
                values = scores[selected]
                rows.append({
                    "Subject": subject, "Subsequent measure": label,
                    "School cutoff": f"{threshold}%", "Scores recorded (n)": len(values),
                    "Mean score": values.mean() if len(values) else np.nan,
                    "Median score": values.median() if len(values) else np.nan,
                    "Observed range": f"{values.min():.1f}-{values.max():.1f}" if len(values) else "none",
                })
    return pd.DataFrame(rows)


def main():
    raw, cbse, pakistan = load_data()
    summaries = all_summaries(cbse)
    later = later_summary(cbse)
    image = plot_data(cbse)
    pakistan_qual = pakistan["Qual / Award"].fillna("(missing)").value_counts().to_dict()
    math60 = summaries[(summaries.Subject == "Mathematics") & (summaries["School mark cutoff"] == "60%")].iloc[0]
    phys60 = summaries[(summaries.Subject == "Physics") & (summaries["School mark cutoff"] == "60%")].iloc[0]

    intro = (f"The workbook has {len(raw)} records and {len(cbse)} with exact qualification label CBSE. "
             f"It has no BISE qualification label. The {len(pakistan)} Pakistani-nationality students have qualification labels: "
             + ", ".join(f"{name} ({count})" for name, count in pakistan_qual.items())
             + ". Nationality is not used as a proxy for school board.")
    recommendation = ("Of the three candidates, use 75% as a provisional minimum eligibility threshold for a monitored exemption pilot, "
                     "not as a validated cutoff. Available data cannot establish that 70% or 75% is safe because placement outcomes among "
                     "higher-mark students are nearly absent. Do not use 60%: the recorded 60%+ tested subsets show weak pass rates, "
                     "particularly in Mathematics. This conclusion applies to the CBSE cohort only; Pakistani BISE students are not identifiable.")

    md = [
        "# Analysis 2: CBSE and Pakistani BISE students", "",
        f"**Recommendation:** {recommendation}", "",
        "## Population and source fields", intro, "",
        "There is no explicit overall secondary-school percentage field. `Math Mark` and `Phys Mark` are treated as subject-specific school percentages on a 0-100 scale. `Overall Score` is not used because it is on a different scale. Mathematics and Physics exemption decisions are evaluated separately.", "",
        "## Evaluation method",
        f"The candidate rules are applied to the relevant subject mark: 75%, 70%, and 60%. Placement pass is provisionally defined as PT score >= {PLACEMENT_PASS_MARK}; the source workbook does not document the official pass mark. Placement-pass proportions use only records with a test score, with Wilson 95% confidence intervals. They do not estimate pass rates for every eligible student because testing is sparse and non-random.", "",
        "Subsequent performance is described using Math 042 Module 1 (out of 100), Math 141 Midterm (out of 100), and Physics PHY 040 Module 1 (out of 50), where recorded. The workbook provides no course-performance pass standard, so no pass/fail classification is assigned to these outcomes.", "",
        "## Placement test evidence", markdown_table(summaries, ["Placement coverage", "PT pass rate among tested", "PT pass rate below cutoff"]), "",
        "The pass-rate comparison below each cutoff is likewise limited to tested students. Missing test scores are neither passes nor failures. Higher-mark groups have almost no test data, making the comparison especially vulnerable to selection bias.", "",
        "## Visual evidence", "The vertical lines mark the candidate school-mark cutoffs; the horizontal line is the assumed placement pass score. The plots are embedded in the [HTML report](../out/analysis_02_cbse_bise.html).", "",
        "## Subsequent performance", markdown_table(later), "",
        "Course scores cover only students with recorded course outcomes. No CBSE Mathematics 042 Module 1 scores are present. Physics results and Math 141 midterm results are available only for subsets. Because students with lower marks may be more likely to enrol in foundation courses, these groups are selected and cannot establish how exempt students would have performed.", "",
        "## Threshold assessment",
        "- **75%:** Most conservative candidate and preferred only as a temporary, monitored pilot threshold. The records do not validate it: placement outcomes for eligible higher-mark students are nearly absent.",
        "- **70%:** Not supported over 75%; there are too few additional observations in the 70-74% band to establish safety.",
        f"- **60%:** Not recommended. Among students at/above 60% with recorded placement tests, Mathematics pass rate is {math60['PT pass rate among tested']:.0%} (n={int(math60['Placement scores recorded (n)'])}); Physics is {phys60['PT pass rate among tested']:.0%} (n={int(phys60['Placement scores recorded (n)'])}). These are selected tested subsets, but they do not support a broad 60% exemption.", "",
        "**Operational recommendation:** If one candidate must be selected now, use a temporary 75% minimum in the relevant subject, subject to faculty approval and prospective monitoring. Keep subject decisions independent. Retain placement testing until test and subsequent-course outcomes are collected consistently across the 60-80% range, including students otherwise eligible for exemption.", "",
        "## Evidence and limitations",
        "- Exact `Qual / Award` label `CBSE` identifies the analyzed CBSE cohort. No row is explicitly identified as BISE. Pakistani nationality alone does not establish BISE education.",
        "- Source fields: `Qual / Award`, `Nationality`, `Math Mark`, `Phys Mark`, `PT MATH`, `PT PHYS`, `Math 042 Module one results out of 100`, `Math 141 Midterm Exam (out of 100)`, and `Phy040 Module 1 Exam (out of 50)`.",
        "- A text value such as `50CT` is parsed as 50. Other non-numeric values are excluded from numerical calculations.",
        "- Placement scores, school marks, and later outcomes have substantial missingness and are not observed uniformly by mark range. Selection bias is likely.",
        "- The placement pass mark, course pass criteria, and target acceptable exemption risk are not specified in the workbook. The 50-point placement pass mark is an explicit assumption.",
        "- Therefore 75% is a cautious provisional choice among the proposed options, not a statistically confirmed minimum.", "",
        "## Reproducibility", "Run `uv run python src/analysis_02_cbse_bise.py`. This recreates this document and `out/analysis_02_cbse_bise.html`.",
    ]
    MD_OUT.parent.mkdir(exist_ok=True)
    MD_OUT.write_text("\n".join(md) + "\n", encoding="utf-8")

    css = """body{font:16px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1160px;margin:32px auto;padding:0 22px;color:#202b36}h1{font-size:2rem;border-bottom:3px solid #2166ac;padding-bottom:10px}h2{margin-top:36px;color:#243b53}.lead,.warning{padding:16px 20px;border-left:5px solid #2166ac;background:#eef5fb}.warning{border-color:#d18b00;background:#fff7e6}.data-table{border-collapse:collapse;margin:12px 0 24px;max-width:100%;font-size:.92rem}.data-table th{background:#243b53;color:white;text-align:left}.data-table th,.data-table td{padding:8px 10px;border:1px solid #d8e0e8}.data-table tr:nth-child(even){background:#f3f6f8}img{width:100%;height:auto}.meta{color:#536575}code{background:#edf0f2;padding:2px 4px;border-radius:3px}li{margin:7px 0}@media(max-width:700px){body{margin:18px auto;padding:0 12px}.data-table{font-size:.78rem}.data-table th,.data-table td{padding:5px}}"""
    plot = f'<img alt="School mark versus placement score scatter plots" src="data:image/png;base64,{image}">'
    warning = html.escape(intro)
    report = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Analysis 2: CBSE and Pakistani BISE</title><style>{css}</style></head><body>
<h1>Analysis 2: CBSE and Pakistani BISE students</h1><p class="meta">Source: {html.escape(DATA.name)} | {len(cbse)} exact-CBSE records | Candidate cutoffs: 75%, 70%, 60%</p>
<div class="lead"><b>Recommendation:</b> {html.escape(recommendation)}</div>
<div class="warning"><b>Population caveat:</b> {warning}</div>
<h2>Method</h2><p>The workbook has no explicit overall school percentage. <code>Math Mark</code> and <code>Phys Mark</code> are analyzed as subject-specific percentages; the subjects are handled independently. Placement pass is provisionally PT score &ge; {PLACEMENT_PASS_MARK}; no official pass mark is documented. Rates and confidence intervals use only students with recorded placement scores. Later course results are reported descriptively because no course pass standard is provided.</p>
<h2>Placement test evidence by proposed cutoff</h2>{html_table(summaries, ["Placement coverage", "PT pass rate among tested", "PT pass rate below cutoff"])}
<p>A missing placement score is not a pass or failure. Higher-mark students are almost entirely absent from the tested subset, so placement results cannot validate 70% or 75% exemptions.</p>{plot}
<h2>Subsequent course performance</h2>{html_table(later)}<p>No CBSE Math 042 Module 1 results are recorded. Physics module results and Math 141 midterm results cover only subsets; these selected outcomes cannot establish exempt-student readiness. Score ranges and denominators are in the Markdown findings.</p>
<h2>Assessment of the three thresholds</h2><ul><li><b>75%:</b> most conservative candidate; use only as a temporary, monitored pilot. Not validated by current outcome data.</li><li><b>70%:</b> not supported over 75%; the 70-74% band has too little placement evidence.</li><li><b>60%:</b> not recommended. At/above 60%, Math pass rate among tested is {math60['PT pass rate among tested']:.0%} (n={int(math60['Placement scores recorded (n)'])}); Physics is {phys60['PT pass rate among tested']:.0%} (n={int(phys60['Placement scores recorded (n)'])}).</li></ul>
<p><b>Operational recommendation:</b> If choosing now, set a temporary 75% minimum in each relevant subject, subject to faculty approval and monitoring. Keep Math and Physics decisions independent. Retain testing until outcomes are captured consistently across the 60-80% range, including students otherwise eligible for exemption.</p>
<h2>Limitations</h2><ul><li>No BISE qualification is identified; nationality is not a reliable proxy for school board.</li><li>Missingness and non-random test/course observation create selection bias.</li><li>The official placement pass mark and course pass criteria are not provided; 50 is a stated assumption for placement tests.</li><li>Full findings and evidence tables: <code>doc/02.md</code>. Rerun with <code>uv run python src/analysis_02_cbse_bise.py</code>.</li></ul></body></html>"""
    HTML_OUT.parent.mkdir(exist_ok=True)
    HTML_OUT.write_text(report, encoding="utf-8")
    print(f"Wrote {HTML_OUT}")
    print(f"Wrote {MD_OUT}")
    print(f"CBSE records: {len(cbse)}; Pakistani-nationality records (not assumed BISE): {len(pakistan)}")
    print(summaries.to_string(index=False))
    print(later.to_string(index=False))


if __name__ == "__main__":
    main()