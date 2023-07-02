#!/usr/bin/env ipython
import argparse
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

def handle_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate an ipython notebook")
    parser.add_argument("execpath", help="Working directory to use during evaluation")
    parser.add_argument("notebook", help="notebook to parse")
    parser.add_argument("--retain_x11", help="Whether to retain X11 for the tests", action='store_true')
    return parser.parse_args()

def run_notebook(notebook_path, exec_path, retain_x11):
    # Read notebook file
    with open(notebook_path) as file:
        notebook = nbformat.read(file, as_version=4)

    # Set environment variables
    if "DISPLAY" in os.environ and not retain_x11:
        del os.environ["DISPLAY"]
    os.environ["MPLCONFIG"] = "ps"

    # Prepare preprocessing engine
    preprocessor = ExecutePreprocessor(timeout=6000)

    # Run notebook
    print("Running notebook")
    preprocessor.preprocess(notebook, {'metadata': {'path': exec_path}})
    print("Preprocess done")

def main():
    # Handle command line arguments
    cfg = handle_args()

    # Run the notebook
    run_notebook(cfg.notebook, cfg.execpath, cfg.retain_x11)

if __name__ == "__main__":
    main()