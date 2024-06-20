#!/usr/bin/env ipython
import argparse
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

def handle_args() -> argparse.Namespace:
    """
    Handle command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate an ipython notebook")
    parser.add_argument("execpath", help="Working directory to use during evaluation")
    parser.add_argument("notebook", help="notebook to parse")
    parser.add_argument("--retain_x11", help="Whether to retain X11 for the tests", action='store_true')
    return parser.parse_args()

def run_notebook(cfg: argparse.Namespace) -> None:
    """
    Run the ipython notebook.

    Args:
        cfg (argparse.Namespace): Parsed command line arguments.
    """
    if "DISPLAY" in os.environ and not cfg.retain_x11:
        del os.environ["DISPLAY"]
    os.environ["MPLCONFIG"] = "ps"

    with open(cfg.notebook) as f:
        nb = nbformat.read(f, as_version=4)

    # Prepare preprocessing engine
    ep = ExecutePreprocessor(timeout=6000)

    # Run it
    print("Running notebook")
    ep.preprocess(nb, {'metadata': {'path': cfg.execpath}})
    print("Preprocess done")

def main() -> None:
    """
    Main function to run the program.
    """
    cfg = handle_args()
    run_notebook(cfg)

if __name__ == "__main__":
    main()