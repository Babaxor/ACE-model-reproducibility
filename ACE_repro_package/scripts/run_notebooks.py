#!/usr/bin/env python
"""
Run all notebooks programmatically to ensure they execute without errors.
"""

import os
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter

def run_notebook(notebook_path):
    """Execute a notebook and save output."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Execute notebook
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
    
    # Save executed notebook
    output_path = notebook_path.replace('.ipynb', '_executed.ipynb')
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    # Also export to HTML for easy viewing
    html_exporter = HTMLExporter()
    html_data, _ = html_exporter.from_notebook_node(nb)
    html_path = notebook_path.replace('.ipynb', '_executed.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_data)
    
    print(f"Executed: {notebook_path}")
    print(f"  Saved to: {output_path}")
    print(f"  HTML version: {html_path}")
    
    return nb

def run_all():
    """Run all notebooks in the notebooks directory."""
    notebook_dir = "../notebooks"
    notebooks = [
        "01_LU_derivation.ipynb",
        "02_Jacobian_analysis.ipynb", 
        "03_asymmetric_verification.ipynb"
    ]
    
    for nb_name in notebooks:
        nb_path = os.path.join(notebook_dir, nb_name)
        if os.path.exists(nb_path):
            print(f"\n{'='*60}")
            print(f"Running notebook: {nb_name}")
            print('='*60)
            try:
                run_notebook(nb_path)
            except Exception as e:
                print(f"ERROR executing {nb_name}: {e}")
        else:
            print(f"Notebook not found: {nb_path}")

if __name__ == "__main__":
    run_all()