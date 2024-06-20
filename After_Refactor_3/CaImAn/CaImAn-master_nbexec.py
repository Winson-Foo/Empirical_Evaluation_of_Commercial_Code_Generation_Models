#!/usr/bin/env ipython
import argparse
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

def handle_args():
    """
    Parse command line arguments and return the configuration.
    """
    parser = argparse.ArgumentParser(description="Evaluate an ipython notebook")
    parser.add_argument("execpath", help="Working directory to use during evaluation")
    parser.add_argument("notebook", help="notebook to parse")
    parser.add_argument("--retain_x11", help="Whether to retain X11 for the tests", action='store_true')
    
    return parser.parse_args()

def run_notebook(notebook_path, exec_path, retain_x11):
    """
    Execute an ipython notebook.
    
    Args:
    - notebook_path: Path to the notebook file.
    - exec_path: Working directory to use during evaluation.
    - retain_x11: Whether to retain X11 for the tests.
    """
    if "DISPLAY" in os.environ and not retain_x11:
        del os.environ["DISPLAY"]
    
    os.environ["MPLCONFIG"] = "ps"
    
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    # Prepare preprocessing engine
    ep = ExecutePreprocessor(timeout=6000)
    
    # Run the notebook
    print("Running notebook")
    ep.preprocess(nb, {'metadata': {'path': exec_path}})
    print("Preprocess done")

def main():
    """
    Main entry point of the script.
    """
    cfg = handle_args()
    run_notebook(cfg.notebook, cfg.execpath, cfg.retain_x11)

if __name__ == "__main__":
    main()