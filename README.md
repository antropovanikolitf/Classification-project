# Wine Classification — Session Demo (Notebooks 01–02)

**Goal:** Binary classification of wine **red vs white** using the UCI Wine Quality chemistry data.
**Today's scope:** Only the first two notebooks (problem framing + data understanding). No modeling yet.

## Contents
- `notebooks/01_problem_framing.ipynb` — problem definition, scope, impact framing, stakeholders.
- `notebooks/02_data_understanding.ipynb` — EDA, class balance, feature distributions, early preprocessing notes.

## Data
UCI Wine Quality (Vinho Verde, Portugal, 2004–2007). Laboratory measurements (pH, alcohol, sulphates, etc.).
Files: `data/winequality-red.csv`, `data/winequality-white.csv` (semicolon-separated).

## How to run
```bash
pip install -r requirements.txt
jupyter notebook
```