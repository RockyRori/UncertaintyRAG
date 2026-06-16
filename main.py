"""
Canonical local entry point for the TPAMI prototype.

This intentionally replaces the old demo script that returned gold answers.
Use `python main.py --max-samples 100` for a governed decision-loop run.
"""

from run_decision_loop import main


if __name__ == "__main__":
    main()
