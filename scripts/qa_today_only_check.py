#!/usr/bin/env python3
"""
QA checker for "today-only" demo (Notebooks 01–02) in the Classification Project.
Usage:
    python scripts/qa_today_only_check.py --root .
Exits with non-zero status if any required check fails.
"""
import os
import sys
import argparse
import re
from pathlib import Path
from datetime import datetime

# Optional imports
try:
    import nbformat
except Exception:
    nbformat = None

try:
    import pandas as pd
except Exception:
    pd = None


PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


REQUIRED_FILES = [
    "notebooks/01_problem_framing.ipynb",
    "notebooks/02_data_understanding.ipynb",
    "data/winequality-red.csv",
    "data/winequality-white.csv",
    "README.md",
    "requirements.txt",
    ".gitignore",
]

GITIGNORE_REQUIRED_PATTERNS = [
    r"\.ipynb_checkpoints/",
    r"venv/",
    r"__pycache__/",
    r"results/",
    r"notebooks/03_.*",
    r"notebooks/04_.*",
    r"notebooks/05_.*",
]

REQUIREMENTS_MIN = {"pandas", "numpy", "scikit-learn", "matplotlib", "seaborn"}

SKLEARN_CLASS_NAMES = [
    "LogisticRegression", "RandomForest", "RandomForestClassifier",
    "GradientBoosting", "GradientBoostingClassifier", "SVC", "KNeighborsClassifier",
    "XGBClassifier", "DecisionTreeClassifier", "GaussianNB", "BernoulliNB",
    "MultinomialNB", "LinearSVC", "LinearDiscriminantAnalysis", "QuadraticDiscriminantAnalysis"
]

def load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:
            return ""

def check_files_exist(root: Path):
    results = []
    for rel in REQUIRED_FILES:
        p = root / rel
        ok = p.exists()
        results.append((f"Exists: {rel}", PASS if ok else FAIL, "" if ok else f"Missing {rel}"))
    return results

def check_gitignore(root: Path):
    p = root / ".gitignore"
    if not p.exists():
        return [("gitignore present & includes rules", FAIL, ".gitignore not found")]
    text = load_text(p)
    missing = [pat for pat in GITIGNORE_REQUIRED_PATTERNS if re.search(pat, text, re.MULTILINE) is None]
    if missing:
        return [("gitignore includes today-only rules", WARN, f"Add patterns: {', '.join(missing)}")]
    return [("gitignore includes today-only rules", PASS, "")]

def check_requirements(root: Path):
    p = root / "requirements.txt"
    if not p.exists():
        return [("requirements.txt present", FAIL, "requirements.txt not found")]
    text = load_text(p)
    found = {line.strip().split("==")[0].lower() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")}
    missing = {pkg.lower() for pkg in REQUIREMENTS_MIN if pkg.lower() not in found}
    if missing:
        return [("requirements includes minimal deps", WARN, f"Missing: {', '.join(sorted(missing))}")]
    return [("requirements includes minimal deps", PASS, "")]

def load_notebook(path: Path):
    if nbformat is None:
        return None, "nbformat not installed (pip install nbformat)"
    try:
        nb = nbformat.read(path, as_version=4)
        return nb, ""
    except Exception as e:
        return None, str(e)

def nb_all_cells(nb):
    for cell in nb.cells:
        yield cell

def nb_markdown_cells(nb):
    for cell in nb.cells:
        if cell.get("cell_type") == "markdown":
            yield cell

def nb_code_cells(nb):
    for cell in nb.cells:
        if cell.get("cell_type") == "code":
            yield cell

def has_outputs(nb):
    for cell in nb_code_cells(nb):
        outs = cell.get("outputs") or []
        if len(outs) > 0:
            return True
    return False

def count_image_outputs(nb):
    cnt = 0
    for cell in nb_code_cells(nb):
        for out in cell.get("outputs") or []:
            data = out.get("data") or {}
            if "image/png" in data or "image/jpeg" in data or "image/svg+xml" in data:
                cnt += 1
    return cnt

def find_strings_in_nb(nb, patterns, search_in=("markdown","code")):
    found = set()
    for cell in nb_all_cells(nb):
        ct = cell.get("cell_type")
        if ct not in search_in:
            continue
        src = cell.get("source") or ""
        for pat in patterns:
            if re.search(pat, src, re.IGNORECASE):
                found.add(pat)
    return found

def check_nb01(root: Path):
    path = root / "notebooks/01_problem_framing.ipynb"
    nb, err = load_notebook(path)
    if nb is None:
        return [("Notebook 01 loads", FAIL, f"Cannot open: {err}")]
    results = [("Notebook 01 loads", PASS, "")]
    must_text = [
        r"Wine Classification",
        r"Problem Framing",
        r"UCI.*Wine Quality|Wine Quality.*UCI",
        r"red\s*vs\.?\s*white|red.*white",
        r"stakeholder|impact|ethic|sustainab",
    ]
    found = find_strings_in_nb(nb, must_text, search_in=("markdown","code"))
    missing = [m for m in must_text if m not in found]
    if missing:
        results.append(("NB01: Problem framing content present", WARN, f"Consider adding: {missing}"))
    else:
        results.append(("NB01: Problem framing content present", PASS, ""))
    repro_hints = [r"__version__", r"random_state\s*=\s*42"]
    found2 = find_strings_in_nb(nb, repro_hints, search_in=("code",))
    if not found2:
        results.append(("NB01: Repro cell (versions/seed)", WARN, "Add a cell that prints pandas/sklearn versions and sets random_state=42 where relevant."))
    else:
        results.append(("NB01: Repro cell (versions/seed)", PASS, ""))
    if has_outputs(nb):
        results.append(("NB01: Saved with outputs", PASS, ""))
    else:
        results.append(("NB01: Saved with outputs", WARN, "Execute and save so GitHub renders outputs."))
    return results

def check_nb02(root: Path):
    path = root / "notebooks/02_data_understanding.ipynb"
    nb, err = load_notebook(path)
    if nb is None:
        return [("Notebook 02 loads", FAIL, f"Cannot open: {err}")]
    results = [("Notebook 02 loads", PASS, "")]
    must_code = [r"read_csv", r"sep\s*=\s*[\"']\;[\"']", r"(?:\['type'\]|type)\s*=", r"concat\("]
    found_code = find_strings_in_nb(nb, must_code, search_in=("code",))
    missing = [m for m in must_code if m not in found_code]
    if missing:
        results.append(("NB02: Data load + label (0/1) + concat", WARN, f"Missing patterns: {missing}"))
    else:
        results.append(("NB02: Data load + label (0/1) + concat", PASS, ""))
    class_vis_hints = [r"value_counts\(", r"countplot", r"barh?\(", r"plot\(", r"hist\("]
    found_vis = find_strings_in_nb(nb, class_vis_hints, search_in=("code",))
    img_count = count_image_outputs(nb)
    if not found_vis or img_count < 1:
        results.append(("NB02: Class balance figure + output", WARN, "Add a bar/plot of class counts and ensure outputs are saved."))
    else:
        results.append(("NB02: Class balance figure + output", PASS, f"Detected {img_count} image outputs."))
    if img_count >= 2:
        results.append(("NB02: >=2 feature visuals present", PASS, f"Detected {img_count} image outputs."))
    else:
        results.append(("NB02: >=2 feature visuals present", WARN, "Add at least two feature plots (e.g., boxplots/hist by class)."))
    interp_markdowns = 0
    # Check markdown cells
    for cell in nb_markdown_cells(nb):
        text = (cell.get("source") or "").lower()
        if any(w in text for w in ["interpretation", "insight", "insights"]):
            if len(text.split()) >= 20:
                interp_markdowns += 1
    # Also check code cells for print statements with interpretations
    for cell in nb_code_cells(nb):
        src = (cell.get("source") or "").lower()
        if any(w in src for w in ["interpretation", "insight", "insights"]):
            interp_markdowns += 1
    if interp_markdowns >= 2:
        results.append(("NB02: Figure interpretations present", PASS, f"{interp_markdowns} interpretation notes found."))
    else:
        results.append(("NB02: Figure interpretations present", WARN, "Add 2–3 line interpretations under figures (use the word 'Interpretation' to pass this check)."))
    code_text = "\n".join([cell.get("source") or "" for cell in nb_code_cells(nb)])
    trained = False
    if re.search(r"\.fit\s*\(", code_text):
        trained = True
    for name in SKLEARN_CLASS_NAMES:
        if name in code_text:
            trained = True
            break
    if trained:
        results.append(("NB02: No model training appears", FAIL, "Detected model training patterns ('.fit(' or classifier names). Move training to Notebook 03."))
    else:
        results.append(("NB02: No model training appears", PASS, ""))
    if has_outputs(nb):
        results.append(("NB02: Saved with outputs", PASS, ""))
    else:
        results.append(("NB02: Saved with outputs", FAIL, "Run all cells and save notebook with outputs so GitHub renders figures."))
    return results

def check_data_shapes(root: Path):
    if pd is None:
        return [("Data shapes (red/white)", WARN, "pandas not available; cannot verify shapes")]
    results = []
    red_p = root / "data/winequality-red.csv"
    white_p = root / "data/winequality-white.csv"
    try:
        red = pd.read_csv(red_p, sep=";")
        white = pd.read_csv(white_p, sep=";")
        ok_red = (1500 <= len(red) <= 1700) and (11 <= red.shape[1] <= 13)
        ok_white = (4700 <= len(white) <= 5100) and (11 <= white.shape[1] <= 13)
        status = PASS if (ok_red and ok_white) else WARN
        msg = f"red={red.shape}, white={white.shape}"
        results.append(("Data shapes (approx UCI)", status, msg))
    except Exception as e:
        results.append(("Data shapes (approx UCI)", FAIL, f"Error reading CSVs: {e}"))
    return results

def summarize(results):
    max_label = max(len(r[0]) for r in results)
    max_status = max(len(r[1]) for r in results)
    lines = []
    failures = 0
    warnings = 0
    for label, status, detail in results:
        if status == FAIL:
            failures += 1
        elif status == WARN:
            warnings += 1
        lines.append(f"{label.ljust(max_label)}  {status.ljust(max_status)}  {detail}")
    header = f"QA Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    sep = "-" * len(header)
    body = "\n".join(lines)
    footer = f"\nSummary: {failures} FAIL, {warnings} WARN\n"
    return f"{header}\n{sep}\n{body}{footer}", failures

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Project root folder (default: .)")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    results = []
    results += check_files_exist(root)
    results += check_gitignore(root)
    results += check_requirements(root)
    results += check_nb01(root)
    results += check_nb02(root)
    results += check_data_shapes(root)

    report, failures = summarize(results)
    print(report)
    sys.exit(1 if failures > 0 else 0)

if __name__ == "__main__":
    main()