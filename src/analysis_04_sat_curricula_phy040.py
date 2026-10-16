"""Analysis 4: PHY 040 outcomes for selected curricula with SAT Math below 600."""
import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Engineering_Student_Profile_with_Exam_Results.xlsx"
HTML_OUT = ROOT / "out" / "analysis_04_sat_curricula_phy040.html"
MD_OUT = ROOT / "doc" / "04.md"
SAT_CUTOFF = 600
TARGET = {
    "american curriculum": "American Curriculum",
    "sabis": "SABIS",
    "al-thanawiyya al-aama": "Al-Thanawiyya Al-Aama",
}


def load_data():
    raw = pd.read_excel(DATA, sheet_name="Engineering Enrollment")
    raw["curriculum_key"] = (
        raw["Qual / Award"].fillna("").astype(str).str.strip().str.casefold().str.replace(r"\s+", " ", regex=True)
    )
    raw["sat_math_score"] = pd.to_numeric(raw["SAT Math"].replace("-", np.nan), errors="coerce")
    raw["phy_module_score"] = pd.to_numeric(raw["Phy040 Module 1 Exam (out of 50)"], errors="coerce")
    raw["phy_pt_score"] = pd.to_numeric(
        raw["PT PHYS"].astype(str).str.extract(r"^\s*(\d+(?:\.\d+)?)")[0], errors="coerce"
    )
    return raw


def build_tables(raw):
    target = raw[raw["curriculum_key"].isin(TARGET)].copy()
    below = target[target["sat_math_score"].lt(SAT_CUTOFF)].copy()
    has_math = below["Take MATH042"].astype(str).str.strip().str.upper().eq("MATH042")
    has_physics = below["Take PHYS040"].astype(str).str.strip().str.upper().isin({"PHYS040", "PHY040"})
    assigned = below[has_math & has_physics].copy()

    eligibility_rows = []
    for key, label in TARGET.items():
        group = target[target["curriculum_key"].eq(key)]
        low = group[group["sat_math_score"].lt(SAT_CUTOFF)]
        both = low[
            low["Take MATH042"].astype(str).str.strip().str.upper().eq("MATH042")
            & low["Take PHYS040"].astype(str).str.strip().str.upper().isin({"PHYS040", "PHY040"})
        ]
        eligibility_rows.append({
            "Curriculum": label,
            "All records (n)": len(group),
            "Numeric SAT Math (n)": int(group["sat_math_score"].notna().sum()),
            "SAT Math missing (n)": int(group["sat_math_score"].isna().sum()),
            "SAT Math < 600 (n)": len(low),
            "Both courses assigned (n)": len(both),
            "PHY 040 module scores in full curriculum (n)": int(group["phy_module_score"].notna().sum()),
        })

    outcome_rows = []
    for key, label in TARGET.items():
        group = assigned[assigned["curriculum_key"].eq(key)]
        scores = group["phy_module_score"].dropna()
        pt = group["phy_pt_score"].dropna()
        enrolled = group["Enrolled in PHYS040"]
        if len(scores) >= 2:
            se = scores.std(ddof=1) / np.sqrt(len(scores))
            half = stats.t.ppf(0.975, len(scores) - 1) * se
            ci_low, ci_high = max(0.0, scores.mean() - half), min(50.0, scores.mean() + half)
        else:
            ci_low = ci_high = np.nan
        outcome_rows.append({
            "Curriculum": label,
            "SAT<600 and both assigned (n)": len(group),
            "Module scores (n)": len(scores),
            "Module coverage": len(scores) / len(group) if len(group) else np.nan,
            "Mean PHY 040 Module 1 (/50)": scores.mean() if len(scores) else np.nan,
            "95% mean CI lower": ci_low,
            "95% mean CI upper": ci_high,
            "Median (/50)": scores.median() if len(scores) else np.nan,
            "SD": scores.std(ddof=1) if len(scores) > 1 else np.nan,
            "Min (/50)": scores.min() if len(scores) else np.nan,
            "Max (/50)": scores.max() if len(scores) else np.nan,
            "Physics PT scores (n)": len(pt),
            "Mean Physics PT score": pt.mean() if len(pt) else np.nan,
            "Enrolled Yes (n)": int(enrolled.eq("Yes").sum()),
            "Enrolled No (n)": int(enrolled.eq("No").sum()),
        })
    return target, below, assigned, pd.DataFrame(eligibility_rows), pd.DataFrame(outcome_rows)


def render_html_table(frame):
    shown = frame.copy()
    for col in shown.columns:
        if shown[col].dtype.kind == "f":
            shown[col] = shown[col].map(lambda x: "n/a" if pd.isna(x) else f"{x:.1f}")
    return shown.to_html(index=False, classes="table", border=0, escape=True)


def render_md_table(frame):
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                value = "n/a" if pd.isna(value) else f"{value:.1f}"
            vals.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def plot_scores(assigned):
    labels, values = [], []
    for key, label in TARGET.items():
        scores = assigned.loc[assigned["curriculum_key"].eq(key), "phy_module_score"].dropna().to_numpy()
        if len(scores):
            labels.append(f"{label} (n={len(scores)})")
            values.append(scores)
    fig, ax = plt.subplots(figsize=(9, 5.3))
    positions = np.arange(1, len(labels) + 1)
    palette = ["#267c78", "#d28c27", "#5279a8"]
    for i, (label, scores) in enumerate(zip(labels, values)):
        jitter = np.linspace(-0.08, 0.08, len(scores)) if len(scores) > 1 else np.array([0.0])
        ax.scatter(scores, positions[i] + jitter, s=75, color=palette[i % len(palette)],
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.plot([scores.mean()], [positions[i]], marker="D", markersize=9, color="#222222", zorder=4)
    ax.set_yticks(positions, labels)
    ax.set_xlim(0, 50)
    ax.set_xlabel("PHY 040 Module 1 Exam score (out of 50); diamonds mark curriculum means")
    ax.set_title("PHY 040 Module 1 results\nSAT Math <600 and both course assignments recorded")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def main():
    raw = load_data()
    target, below, assigned, eligibility, outcomes = build_tables(raw)
    image = plot_scores(assigned)
    american = outcomes[outcomes["Curriculum"].eq(TARGET["american curriculum"])].iloc[0]
    sabis = outcomes[outcomes["Curriculum"].eq(TARGET["sabis"])].iloc[0]
    ath = eligibility[eligibility["Curriculum"].eq(TARGET["al-thanawiyya al-aama"])].iloc[0]
    althaw_module = int(target.loc[target["curriculum_key"].eq("al-thanawiyya al-aama"), "phy_module_score"].notna().sum())
    both_flags = (
        below["Take MATH042"].astype(str).str.strip().str.upper().eq("MATH042")
        & below["Take PHYS040"].astype(str).str.strip().str.upper().isin({"PHYS040", "PHY040"})
    )
    assignment_gaps = below[~both_flags]
    sat_missing_total = int(target["sat_math_score"].isna().sum())

    finding = (
        f"Within the verified SAT Math <{SAT_CUTOFF} cohort with both course assignments recorded, American Curriculum students "
        f"have a mean PHY 040 Module 1 score of {american['Mean PHY 040 Module 1 (/50)']:.2f}/50 "
        f"(n={int(american['Module scores (n)'])}, median {american['Median (/50)']:.1f}, "
        f"range {american['Min (/50)']:.1f}-{american['Max (/50)']:.1f}). SABIS has only "
        f"{int(sabis['Module scores (n)'])} recorded {'score' if int(sabis['Module scores (n)']) == 1 else 'scores'}, so its mean is not a reliable comparison. "
        "Al-Thanawiyya Al-Aama cannot be assessed against SAT<600 because its SAT Math values are missing."
    )

    doc = [
        "# Analysis 4: PHY 040 outcomes for selected curricula with SAT Math below 600", "",
        f"**Finding:** {finding}", "",
        "## Cohort definition and data coverage",
        f"The source is `{DATA.name}`, sheet `Engineering Enrollment`. Curricula are selected by the exact case-insensitive `Qual / Award` labels American Curriculum, SABIS, and Al-Thanawiyya Al-Aama. The target SAT condition is a numeric SAT Math score <{SAT_CUTOFF}. Missing values and `-` are not interpreted as below 600.",
        f"The primary performance cohort additionally requires both `Take MATH042 = MATH042` and `Take PHYS040` equal to `PHYS040` (or the workbook variant `PHY040`). This verifies the stated course requirement in the dataset. That yields {len(assigned)} students. Of these, {int(assigned['phy_module_score'].notna().sum())} have a recorded PHY 040 Module 1 exam score; the remainder are not assigned a score in the workbook.", "",
        "## Eligibility and record-level coverage", render_md_table(eligibility), "",
        f"Across the {len(target)} records in the three curricula, {sat_missing_total} SAT Math values are missing or recorded as `-`. In particular, all {int(ath['All records (n)'])} Al-Thanawiyya Al-Aama records lack a numeric SAT Math score. {althaw_module} students in that full curriculum group have a PHY 040 Module 1 score, but the dataset cannot establish whether they meet the SAT<600 criterion. They are therefore excluded from the primary cohort, not assumed to be eligible or ineligible.",
        f"There are {len(below)} students with known SAT Math <600 across the selected curricula. {len(assignment_gaps)} of these do not have both course assignment flags recorded; they are not included in the primary cohort. This documents that the workbook's assignment fields do not perfectly mirror the stated policy.", "",
        "## PHY 040 Module 1 performance", render_md_table(outcomes), "",
        f"[Open the HTML report with the individual-score visualization](../out/analysis_04_sat_curricula_phy040.html).", "",
        "The mean is reported on the recorded exam scale (out of 50). The available score count is the denominator for each curriculum's descriptive statistics. Confidence intervals are t-intervals for the mean bounded to the possible 0-50 score range; no interval is calculated for a single score.", "",
        "## Observations and recommendation",
        f"- **American Curriculum:** {int(american['Module scores (n)'])} scores are available among {int(american['SAT<600 and both assigned (n)'])} students with SAT<600 and both course assignments. Mean {american['Mean PHY 040 Module 1 (/50)']:.2f}/50, median {american['Median (/50)']:.1f}, range {american['Min (/50)']:.1f}-{american['Max (/50)']:.1f}. Performance varies materially, so SAT Math below 600 does not imply uniformly low PHY 040 performance.",
        f"- **SABIS:** only {int(sabis['Module scores (n)'])} score is present among {int(sabis['SAT<600 and both assigned (n)'])} assigned students. The observed score ({sabis['Mean PHY 040 Module 1 (/50)']:.1f}/50) is not enough to characterize SABIS students.",
        f"- **Al-Thanawiyya Al-Aama:** the curriculum has {int(ath['All records (n)'])} records and {althaw_module} PHY 040 scores in its broader group, but no usable SAT Math values. Its performance cannot be attributed to the under-600 policy cohort.",
        "- **Recommendation:** continue the SAT<600 placement/foundation route provisionally, but do not use these small, incomplete Module 1 results to change the 600 threshold or rank curricula. Capture SAT Math, both assignment decisions, actual enrollment, and Module 1 results consistently for future cohorts; review Al-Thanawiyya records with Admissions because the key eligibility field is absent.", "",
        "## Limitations", "- The report excludes SAT Math values that are missing or marked `-`; it does not impute them.",
        "- The primary cohort requires explicit evidence of both course assignments. Two or more SAT<600 students can lack one or both assignment flags, so policy application should be checked against source records.",
        "- PHY 040 exam results are missing for some students with both assignment flags, and curriculum group sizes are highly uneven.",
        "- No PHY 040 Module 1 pass threshold is specified; no pass/fail rates are inferred.",
        "- These are descriptive outcomes, not causal estimates of curriculum effects or the impact of the SAT threshold.", "",
        "## Reproducibility", "Run `uv run python src/analysis_04_sat_curricula_phy040.py`. This recreates this document and `out/analysis_04_sat_curricula_phy040.html`.",
    ]
    MD_OUT.parent.mkdir(exist_ok=True)
    MD_OUT.write_text("\n".join(doc) + "\n", encoding="utf-8")

    css = """body{font:16px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1120px;margin:32px auto;padding:0 22px;color:#202b36}h1{font-size:2rem;border-bottom:3px solid #287c78;padding-bottom:10px}h2{margin-top:34px;color:#284b55}.lead,.caveat{padding:16px 20px;border-left:5px solid #287c78;background:#edf7f4}.caveat{border-color:#cc8c22;background:#fff7e7}.table{border-collapse:collapse;margin:12px 0 24px;max-width:100%;font-size:.9rem}.table th{background:#284b55;color:white;text-align:left}.table th,.table td{padding:7px 9px;border:1px solid #d8e0e8}.table tr:nth-child(even){background:#f3f6f8}img{width:100%;height:auto}code{background:#edf0f2;padding:2px 4px;border-radius:3px}li{margin:7px 0}@media(max-width:700px){body{margin:18px auto;padding:0 12px}.table{font-size:.75rem}.table th,.table td{padding:5px}}"""
    caveat = (
        f"The primary cohort is n={len(assigned)}; only {int(assigned['phy_module_score'].notna().sum())} have exam scores. "
        f"There are {sat_missing_total} missing SAT Math records across the three curricula, including every Al-Thanawiyya record."
    )
    report = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Analysis 4: PHY 040 by curriculum and SAT Math</title><style>{css}</style></head><body>
<h1>Analysis 4: PHY 040 performance by curriculum</h1><p>Selected American, SABIS, and Al-Thanawiyya Al-Aama curricula; SAT Math below 600; MATH042 and PHY040 assignment recorded.</p>
<div class="lead"><b>Finding:</b> {html.escape(finding)}</div><div class="caveat"><b>Coverage caveat:</b> {html.escape(caveat)}</div>
<h2>Eligibility and assignment evidence</h2><p>The SAT threshold is applied only to numeric values. A missing or dash value is not treated as a score below 600. The primary cohort also requires both course assignment flags, to verify the stated policy in the workbook.</p>{render_html_table(eligibility)}<p>{len(below)} selected-curriculum records have numeric SAT Math below 600; {len(assignment_gaps)} do not have both assignment flags recorded. The core analysis includes {len(assigned)} records with both flags.</p>
<h2>PHY 040 Module 1 outcomes</h2>{render_html_table(outcomes)}<img alt="Individual PHY 040 Module 1 scores by curriculum" src="data:image/png;base64,{image}">
<h2>Observations and recommendations</h2><ul><li><b>American Curriculum:</b> mean {american['Mean PHY 040 Module 1 (/50)']:.2f}/50, median {american['Median (/50)']:.1f}, range {american['Min (/50)']:.1f}-{american['Max (/50)']:.1f}, based on n={int(american['Module scores (n)'])} recorded scores.</li><li><b>SABIS:</b> only n={int(sabis['Module scores (n)'])} module score is recorded, too few to characterize the curriculum.</li><li><b>Al-Thanawiyya Al-Aama:</b> {int(ath['All records (n)'])} records have no numeric SAT Math score. Although {althaw_module} PHY 040 scores exist in the broader curriculum group, their SAT<600 eligibility cannot be established.</li><li><b>Recommendation:</b> do not adjust the SAT 600 rule or rank the curricula based on this limited evidence. Maintain the current route provisionally, collect consistent test, assignment, enrollment, and exam outcomes, and resolve missing SAT Math records for Al-Thanawiyya.</li></ul>
<h2>Limitations</h2><ul><li>{len(assigned)} students meet the primary cohort rule, but only {int(assigned['phy_module_score'].notna().sum())} have Module 1 results.</li><li>Groups are very small and unbalanced; no inferential ranking is warranted.</li><li>No Module 1 pass standard is specified, so results are descriptive and no pass/fail status is inferred.</li><li>Full methods and evidence tables: <code>doc/04.md</code>. Rerun with <code>uv run python src/analysis_04_sat_curricula_phy040.py</code>.</li></ul></body></html>"""
    HTML_OUT.parent.mkdir(exist_ok=True)
    HTML_OUT.write_text(report, encoding="utf-8")
    print(f"Wrote {HTML_OUT}")
    print(f"Wrote {MD_OUT}")
    print(f"Selected curricula n={len(target)}; SAT<600 n={len(below)}; both assignments n={len(assigned)}; module scores n={int(assigned['phy_module_score'].notna().sum())}")
    print(eligibility.to_string(index=False))
    print(outcomes.to_string(index=False))


if __name__ == "__main__":
    main()