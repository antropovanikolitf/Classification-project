#!/usr/bin/env python3
"""Fix notebook format issues"""
import json
import sys

def fix_notebook(path):
    with open(path, 'r') as f:
        nb = json.load(f)

    # Ensure all code cells have outputs
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            if 'outputs' not in cell:
                cell['outputs'] = []
            if 'execution_count' not in cell:
                cell['execution_count'] = None

    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"Fixed: {path}")

if __name__ == "__main__":
    fix_notebook("notebooks/02_data_understanding.ipynb")