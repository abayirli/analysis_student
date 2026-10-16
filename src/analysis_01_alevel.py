"""Analysis 1: A-Level exemption cutoff for the Mathematics and Physics placement tests."""
import base64
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
OUT = ROOT / "out" / "analysis_01_alevel.html"
DOC_OUT = ROOT / "doc" / "01.md"

PASS_MARK = 50  # Assumption: the dataset does not state the placement-test pass mark.
SENSITIVITY_MARKS = [40, 50, 60]
TARGET_PASS_RATE = 0.75  # An exempt grade band must have at least this share passing the PT.
GRADES = ["F", "E", "D", "C", "B", "A", "A*"]
RANK = {g: i for i, g in enumerate(GRADES)}
SUBJECTS = {"Math": ("Grade Math", "PT MATH", "MATH 042"), "Physics": ("Grade Phys", "PT PHYS", "PHY 040")}


def load() -> pd.DataFrame:
    raw = pd.read_excel(DATA, sheet_name=0)
    qual = raw["Qual / Award"].astype(str).str.strip().str.lower()
    df = raw[qual.isin(["a level", "a-level"])].copy()  # excludes AS Level and combined qualifications

    def grade(v):
        v = str(v).strip().upper()
        return v if v in GRADES else np.nan

    def score(v):
        m = re.match(r"^\s*(\d+(?:\.\d+)?)", str(v))  # handles entries such as "50CT"
        return float(m.group(1)) if m else np.nan

    out = pd.DataFrame({"id": df["#"]})
    for key, (gcol, pcol, _) in SUBJECTS.items():
        out[f"{key}_grade"] = df[gcol].map(grade).values
        out[f"{key}_pt"] = df[pcol].map(score).values
    return out.reset_index(drop=True)


def fig_to_html(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" alt="plot">'


def subject_frame(df, key):
    d = df[[f"{key}_grade", f"{key}_pt"]].dropna()
    d.columns = ["grade", "pt"]
    d["rank"] = d["grade"].map(RANK)
    return d


def grade_table(d, mark):
    rows = []
    for g in reversed(GRADES):
        s = d[d["grade"] == g]["pt"]
        if len(s):
            rows.append({"Grade": g, "n": len(s), "Mean PT": s.mean(), "Median PT": s.median(),
                         "Min": s.min(), "Max": s.max(), f"Pass rate (PT >= {mark})": (s >= mark).mean()})
    return pd.DataFrame(rows)


def cutoff_table(d, mark):
    rows = []
    for g in ["B", "C", "D", "E"]:
        ex, req = d[d["rank"] >= RANK[g]], d[d["rank"] < RANK[g]]
        ci = stats.binomtest(int((ex["pt"] >= mark).sum()), len(ex)).proportion_ci(method="wilson") if len(ex) else None
        rows.append({
            "Exempt if grade >=": g,
            "Exempted (n)": len(ex),
            "Exempted who pass PT": (ex["pt"] >= mark).mean() if len(ex) else np.nan,
            "95% CI": f"{ci.low:.0%} - {ci.high:.0%}" if ci else "n/a",
            "Exempted who would fail PT (n)": int((ex["pt"] < mark).sum()),
            "Required to enrol (n)": len(req),
            "Required who would pass PT (n)": int((req["pt"] >= mark).sum()),
        })
    return pd.DataFrame(rows)


def recommend_cutoff(d, mark):
    """Step down from A; accept a lower grade unless its own students (if any) fall below the target pass rate."""
    cutoff = "A"
    for g in ["B", "C", "D"]:
        s = d[d["grade"] == g]["pt"]
        if len(s) and (s >= mark).mean() < TARGET_PASS_RATE:
            break
        cutoff = g
    return cutoff


def pct(x):
    return "n/a" if pd.isna(x) else f"{x:.0%}"


def table_html(df, pct_cols=(), float_cols=()):
    df = df.copy()
    for c in df.columns:
        if c.startswith("Pass rate") or c in pct_cols:
            df[c] = df[c].map(pct)
        elif df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:.1f}")
    return df.to_html(index=False, classes="tbl", border=0, escape=False)


def table_markdown(df, pct_cols=()):
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column in pct_cols:
                value = pct(value)
            elif isinstance(value, (float, np.floating)):
                digits = 3 if "p-value" in column.lower() or "rho" in column.lower() else 1
                value = "n/a" if pd.isna(value) else f"{value:.{digits}f}"
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    df = load()
    data = {k: subject_frame(df, k) for k in SUBJECTS}
    colors = {"A*": "#1b7837", "A": "#5aae61", "B": "#a6dba0", "C": "#fee08b", "D": "#fdae61", "E": "#f46d43", "F": "#d73027"}

    # --- Figures ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (k, (_, _, course)) in zip(axes, SUBJECTS.items()):
        d = data[k]
        present = [g for g in GRADES if g in set(d["grade"])]
        groups = [d[d["grade"] == g]["pt"] for g in present]
        ax.boxplot(groups, tick_labels=[f"{g}\n(n={len(s)})" for g, s in zip(present, groups)], showfliers=False)
        rng = np.random.default_rng(0)
        for i, (g, s) in enumerate(zip(present, groups), start=1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(s)), s, color=colors[g], edgecolor="k", zorder=3, s=40)
        ax.axhline(PASS_MARK, color="red", ls="--", label=f"Assumed pass mark ({PASS_MARK})")
        ax.set(title=f"{k} placement test score by A-Level {k} grade", xlabel="A-Level grade", ylabel="Placement test score")
        ax.legend()
    fig1 = fig_to_html(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (k, _) in zip(axes, SUBJECTS.items()):
        t = grade_table(data[k], PASS_MARK)
        bars = ax.bar(t["Grade"], t[f"Pass rate (PT >= {PASS_MARK})"], color=[colors[g] for g in t["Grade"]], edgecolor="k")
        for b, n in zip(bars, t["n"]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"n={n}", ha="center")
        ax.axhline(TARGET_PASS_RATE, color="grey", ls=":", label=f"Target ({TARGET_PASS_RATE:.0%})")
        ax.set(ylim=(0, 1.15), title=f"{k}: share passing the placement test", xlabel="A-Level grade", ylabel="Pass rate")
        ax.legend(loc="lower right")
    fig2 = fig_to_html(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (k, _) in zip(axes, SUBJECTS.items()):
        t = cutoff_table(data[k], PASS_MARK)
        x = np.arange(len(t))
        ax.bar(x - 0.2, t["Exempted who would fail PT (n)"], 0.4, color="#d73027", label="Exempted but would fail PT")
        ax.bar(x + 0.2, t["Required who would pass PT (n)"], 0.4, color="#4575b4", label="Required but would pass PT")
        ax.set_xticks(x, [f">= {g}" for g in t["Exempt if grade >="]])
        ax.set(title=f"{k}: errors under each cutoff", xlabel="Exemption cutoff", ylabel="Students")
        ax.legend()
    fig3 = fig_to_html(fig)

    # --- Mixed profiles (both grades C/D) ---
    both = df.dropna(subset=["Math_grade", "Physics_grade"])
    combos = {"Grade C (Math C, Phys C)": ("C", "C"), "Grade D (Math D, Phys D)": ("D", "D"),
              "Math C, Phys D": ("C", "D"), "Math D, Phys C": ("D", "C")}
    rows = []
    for label, (mg, pg) in combos.items():
        s = both[(both["Math_grade"] == mg) & (both["Physics_grade"] == pg)]
        mp, pp = s["Math_pt"].dropna(), s["Physics_pt"].dropna()
        rows.append({"Profile": label, "Students (n)": len(s),
                     "Math PT n": len(mp), "Math PT mean": mp.mean(), f"Pass rate Math (PT >= {PASS_MARK})": (mp >= PASS_MARK).mean() if len(mp) else np.nan,
                     "Phys PT n": len(pp), "Phys PT mean": pp.mean(), f"Pass rate Phys (PT >= {PASS_MARK})": (pp >= PASS_MARK).mean() if len(pp) else np.nan})
    mixed = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    mk = {("C", "C"): "o", ("D", "D"): "s", ("C", "D"): "^", ("D", "C"): "v"}
    for (mg, pg), m in mk.items():
        s = both[(both["Math_grade"] == mg) & (both["Physics_grade"] == pg)].dropna(subset=["Math_pt", "Physics_pt"])
        ax.scatter(s["Math_pt"], s["Physics_pt"], marker=m, s=80, edgecolor="k", label=f"Math {mg} / Phys {pg} (n={len(s)})")
    ax.axvline(PASS_MARK, color="red", ls="--")
    ax.axhline(PASS_MARK, color="red", ls="--")
    ax.set(xlabel="Math placement test", ylabel="Physics placement test", title="Mixed C/D profiles (students with both PT scores)")
    ax.legend()
    fig4 = fig_to_html(fig)

    # --- Statistics ---
    stat_rows = []
    for k in SUBJECTS:
        d = data[k]
        rho, p = stats.spearmanr(d["rank"], d["pt"])
        c, dd = d[d["grade"] == "C"]["pt"], d[d["grade"] == "D"]["pt"]
        u = stats.mannwhitneyu(c, dd, alternative="two-sided") if len(c) and len(dd) else None
        stat_rows.append({"Subject": k, "Students (n)": len(d), "Spearman rho (grade vs PT)": rho, "p-value": p,
                          "Median PT, grade C": c.median(), "Median PT, grade D": dd.median(),
                          "Mann-Whitney C vs D p-value": u.pvalue if u else np.nan})
    stat_df = pd.DataFrame(stat_rows)

    # --- Sensitivity to pass mark ---
    sens = []
    for k in SUBJECTS:
        for m in SENSITIVITY_MARKS:
            sens.append({"Subject": k, "Pass mark": m, "Recommended cutoff": recommend_cutoff(data[k], m) or "none"})
    sens_df = pd.DataFrame(sens)

    rec = {k: recommend_cutoff(data[k], PASS_MARK) for k in SUBJECTS}
    # --- Report ---
    def rate(k, g):
        s = data[k][data[k]["grade"] == g]["pt"]
        return (s >= PASS_MARK).mean() if len(s) else np.nan, len(s)

    cutoff_txt = []
    for k, (_, _, course) in SUBJECTS.items():
        r = rec[k]
        below = {"A": "A* only", "B": "B", "C": "C", "D": "D"}[r]
        cutoff_txt.append(f"<li><b>{k}:</b> exempt students with A-Level grade <b>{below if r == 'A' else r + ' or above'}</b>; "
                          f"students below this must enrol directly in <b>{SUBJECTS[k][2]}</b>.</li>")

    c_rates = {k: rate(k, "C") for k in SUBJECTS}
    d_rates = {k: rate(k, "D") for k in SUBJECTS}
    both_n = len(both)

    def fmt(r):
        return f"{pct(r[0])} (n={r[1]})"

    math_grade = grade_table(data["Math"], PASS_MARK)
    physics_grade = grade_table(data["Physics"], PASS_MARK)
    math_cutoffs = cutoff_table(data["Math"], PASS_MARK)
    physics_cutoffs = cutoff_table(data["Physics"], PASS_MARK)
    recommendation = (
        f"Under the assumed PT pass mark of {PASS_MARK} and the {TARGET_PASS_RATE:.0%} step-down rule, "
        "use **B or above as a provisional exemption threshold in each subject**. Students below B should be "
        "placed directly into the corresponding foundation course: MATH 042 for Mathematics and PHY 040 for Physics. "
        "The recommendation is provisional: Mathematics has no tested grade-B students, so its B boundary is not directly validated."
    )
    corr_df = stat_df[["Subject", "Students (n)", "Spearman rho (grade vs PT)", "p-value",
                       "Median PT, grade C", "Median PT, grade D", "Mann-Whitney C vs D p-value"]].copy()
    mixed_md = mixed.copy()
    for col in mixed_md.columns:
        if col.startswith("Pass rate"):
            mixed_md[col] = mixed_md[col].map(pct)
        elif mixed_md[col].dtype.kind == "f":
            mixed_md[col] = mixed_md[col].map(lambda value: "n/a" if pd.isna(value) else f"{value:.1f}")
    cutoff_md = pd.concat([math_cutoffs.assign(Subject="Mathematics"), physics_cutoffs.assign(Subject="Physics")], ignore_index=True)
    cutoff_md = cutoff_md[["Subject", "Exempt if grade >=", "Exempted (n)", "Exempted who pass PT", "95% CI",
                           "Exempted who would fail PT (n)", "Required to enrol (n)", "Required who would pass PT (n)"]]
    markdown = f"""# Analysis 1: A-Level placement-test exemption

{recommendation}

## Population and method

Source: `{DATA.name}`, sheet `Engineering Enrollment`. The analysis includes {len(df)} records whose `Qual / Award` is exactly `A level` or `A Level` (case-insensitive). AS Level and combined qualifications are excluded.

There are {len(data['Math'])} students with both a recognized A-Level Mathematics letter grade and a recorded Math Placement Test score, and {len(data['Physics'])} with both a Physics grade and Physics Placement Test score. Grades are taken from `Grade Math` and `Grade Phys`; scores are from `PT MATH` and `PT PHYS`. The parser accepts numeric values and leading numbers in values such as `50CT`.

The workbook does not state the official Placement Test pass mark. This analysis assumes a score of **{PASS_MARK} or higher is a pass**. It also uses a **{TARGET_PASS_RATE:.0%} grade-band pass-rate target**, a policy assumption rather than a value in the data. Starting at B, the rule steps down one grade at a time and stops at the first grade whose own observed pass rate is below target. Missing grades and missing PT scores are excluded from subject calculations; students previously exempted or not tested may differ systematically from the observed sample.

## Recommendation

{recommendation}

The grade-level evidence is consistent with stopping before C in both subjects: Mathematics C has a {fmt(c_rates['Math'])} pass rate and Physics C has {fmt(c_rates['Physics'])}, each below the {TARGET_PASS_RATE:.0%} target. However, the Mathematics data contains no grade-B PT observations; the three students at or above B are one A and two A* students. Therefore, B is an operational cutoff suggested by the step-down rule, not a directly measured Math-B threshold. Physics has two tested B students, both passing, but this sample is also small.

## Placement scores by grade

### Mathematics

{table_markdown(math_grade, [f"Pass rate (PT >= {PASS_MARK})"])}

### Physics

{table_markdown(physics_grade, [f"Pass rate (PT >= {PASS_MARK})"])}

The observed pass rates among grades C and D are mixed. In Mathematics, C and D each have 2/3 passing scores; in Physics, C has 2/4 and D has 5/6. Physics D's higher observed pass rate than C is non-monotonic and based on small groups, so it does not justify exempting D while requiring C to test.

## Candidate cutoff tradeoffs

For each cutoff, “Exempted who would fail PT” counts tested students who would have been exempted but scored below 50. “Required who would pass PT” counts tested students placed below the cutoff who scored at least 50. The confidence interval is a 95% Wilson interval for the pass rate among the tested exempt group.

{table_markdown(cutoff_md, ["Exempted who pass PT"])}

At the B cutoff, Math has 3/3 tested students pass (all are A/A*; no B grade is represented) and Physics has 4/4 pass. At C, 2 of 9 Math and 2 of 8 Physics students who would be exempted score below the assumed PT pass mark. The small samples yield wide intervals.

## Grade and test relationship

{table_markdown(corr_df)}

Spearman correlation is positive in both subjects, but the samples are small. The exploratory C-versus-D Mann-Whitney tests do not show statistically significant score differences. These statistics do not establish that a grade threshold causes better placement-test performance.

## C/D and mixed-grade profiles

There are {len(both)} students with a recognized grade in both subjects. The four requested C/D combinations are summarized below; PT denominators are the students in that combination with a score in the relevant test.

{table_markdown(mixed_md)}

Recommendation for C, D, and mixed C/D students: do not grant a placement-test exemption based on these grades alone. Under the provisional B-or-above rule, the student is assigned to the foundation course for each subject with a grade below B: Math below B to MATH 042; Physics below B to PHY 040. The sample in the mixed groups is especially small, so apply subject-specific review if official placement-test results or other validated evidence are available.

## Sensitivity to the assumed PT pass mark

{table_markdown(sens_df)}

The suggested cutoff varies when the assumed pass mark changes. At PT pass mark 40, the rule returns C for Math and D for Physics; at 60, it returns B for Math and A for Physics. Confirm the institution's actual pass marks before adopting the recommendation.

## Limitations and reproducibility

- The PT pass mark (50), the target pass rate (75%), and the step-down rule are assumptions, not stated institutional standards.
- Only 13 Math and 15 Physics grade/PT pairs are available. Math has no grade-B test observations, and confidence intervals are wide.
- PT coverage may be selective because students already exempted or not tested are absent from the grade-level score analysis.
- C/D mixed-profile groups contain only 1-11 students, with even fewer PT scores. Recommendations for these combinations are conservative.
- Scores such as `50CT` are treated as 50; nonnumeric values and missing values are not scored.

Run `uv run python src/analysis_01_alevel.py` to regenerate this Markdown and the [HTML report](../out/analysis_01_alevel.html).
"""
    DOC_OUT.parent.mkdir(exist_ok=True)
    DOC_OUT.write_text(markdown, encoding="utf-8")

    sections = f"""
<h1>Analysis 1: A-Level students - placement test exemption cutoff</h1>
<p class="meta">Source: <code>{DATA.name}</code> (sheet "Engineering Enrollment"). Students with qualification "A level"/"A Level": {len(df)}.
AS Level and combined qualifications (e.g. "A Level &amp; Bachelors") are excluded.</p>

<div class="box"><h2>Recommendation</h2>
<ul>{''.join(cutoff_txt)}
<li><b>Grade C (both subjects):</b> do <b>not</b> exempt - require the placement tests. Observed pass rate was {fmt(c_rates['Math'])} in Math and {fmt(c_rates['Physics'])} in Physics, below the {TARGET_PASS_RATE:.0%} target.</li>
<li><b>Grade D (both subjects):</b> do <b>not</b> exempt - require the placement tests. Observed pass rate was {fmt(d_rates['Math'])} in Math and {fmt(d_rates['Physics'])} in Physics; where a D student scores poorly the direct route (MATH 042 / PHY 040) is appropriate.</li>
<li><b>Mixed results (C Math / D Phys, or D Math / C Phys):</b> do <b>not</b> exempt either subject; sit both placement tests and enrol in MATH 042 and/or PHY 040 based on each subject's result. Only {int(mixed['Students (n)'].iloc[2] + mixed['Students (n)'].iloc[3])} such students exist and few have PT scores, so this is a conservative judgement rather than a statistically supported cutoff.</li>
</ul>
<p><b>Rule used:</b> start at A and step down; a grade is exempt if the students holding it passed the placement test at a rate of at least {TARGET_PASS_RATE:.0%}
(PT &ge; {PASS_MARK}); stop at the first grade that fails this test.</p></div>

<div class="warn"><h2>Read this first - data limitations and assumptions</h2><ul>
<li><b>Small sample:</b> only {len(data['Math'])} students have both an A-Level Math grade and a Math PT score, and {len(data['Physics'])} for Physics. The cutoff is indicative; confidence intervals are wide.</li>
<li><b>Pass mark is assumed to be {PASS_MARK}</b> for both tests - the dataset does not state it. See the sensitivity table: the answer changes at a higher pass mark.</li>
<li>Many A-Level students have no letter grade (recorded as "-" or missing) or no placement score; they are excluded from the relevant calculation.</li>
<li>Students already exempted or who did not sit the test cannot be evaluated, so there may be selection bias.</li>
<li>Entries such as "50CT" are read as 50. Numeric Physics grade entries (e.g. 35) and non-letter grades are discarded.</li></ul></div>

<h2>1. Placement test performance by A-Level grade</h2>
{fig1}
<h3>Mathematics</h3>{table_html(grade_table(data['Math'], PASS_MARK))}
<h3>Physics</h3>{table_html(grade_table(data['Physics'], PASS_MARK))}
{fig2}

<h2>2. Candidate cutoffs - consequences</h2>
<p>"Exempted who would fail PT" = students who would skip the test but score below the pass mark (risk of under-placement).
"Required who would pass PT" = students forced to take MATH 042 / PHY 040 despite being able to pass the test (over-placement).</p>
<h3>Mathematics</h3>{table_html(cutoff_table(data['Math'], PASS_MARK))}
<h3>Physics</h3>{table_html(cutoff_table(data['Physics'], PASS_MARK))}
{fig3}

<h2>3. Strength of the grade/test relationship</h2>
{table_html(stat_df)}
<p>A positive Spearman correlation means higher grades tend to score higher in the placement test. Grade C and D students are not distinguishable
from each other on PT score (Mann-Whitney p-values are large), which is why C and D are treated the same way.</p>

<h2>4. Mixed C/D profiles</h2>
<p>{both_n} students have grades in both subjects.</p>
{table_html(mixed)}
{fig4}

<h2>5. Sensitivity to the assumed pass mark</h2>
{table_html(sens_df)}
<p>Cutoff grade shown is the lowest grade exempt under the stepwise rule. Raising the pass mark does not make the cutoff more lenient.</p>
"""
    css = """body{font-family:Arial,sans-serif;max-width:1100px;margin:30px auto;padding:0 16px;color:#222}
h1{border-bottom:3px solid #4575b4;padding-bottom:8px}h2{margin-top:36px;color:#333}.meta{color:#555}
.tbl{border-collapse:collapse;margin:10px 0}.tbl th{background:#4575b4;color:#fff;padding:6px 10px}.tbl td{padding:5px 10px;border-bottom:1px solid #ddd;text-align:center}
.box{background:#eef5ff;border-left:5px solid #4575b4;padding:6px 18px;margin:20px 0}.warn{background:#fff6e5;border-left:5px solid #f0a020;padding:6px 18px;margin:20px 0}
img{max-width:100%}code{background:#eee;padding:1px 4px}"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Analysis 1 - A-Level</title><style>{css}</style></head><body>{sections}</body></html>", encoding="utf-8")
    print(f"Report written to {OUT}")
    print(f"Findings written to {DOC_OUT}")
    return rec


if __name__ == "__main__":
    print(main())
