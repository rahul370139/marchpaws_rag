"""Command‑line demo for the MARCH‑PAWS RAG assistant.

This script shows how to use the `Orchestrator` class to handle a single
interactive session.  It accepts a free‑text query and executes a number of
steps of the MARCH‑PAWS protocol, printing the checklist and citations for
each phase.

Example:

```bash
python examples/demo.py --query "Gunshot wound to right chest, SPO2 85%, tracheal shift" \
  --n-steps 4
```

Note: Before running this demo you must build the BM25 index with
`build_index.py` and ensure that a compatible LLM server is running at
`http://localhost:11434/api/generate`.
"""

import argparse
from pathlib import Path
import sys

# Ensure that the src directory is on the Python path.  This allows the example
# to be run directly via `python examples/demo.py` without installing the
# package.  When installed as a package the following is unnecessary.
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from orchestrator import Orchestrator


def main():
    parser = argparse.ArgumentParser(description="Run an example MARCH‑PAWS session.")
    parser.add_argument("--query", required=True, help="User query describing the casualty's condition.")
    parser.add_argument("--bm25", default="data/anchored_bm25_index.pkl", help="Path to BM25 index pickle.")
    parser.add_argument("--faiss", default="data/anchored_faiss.index", help="Path to FAISS index file (optional).")
    parser.add_argument("--n-steps", type=int, default=3, help="Number of MARCH‑PAWS states to run.")
    args = parser.parse_args()
    if not Path(args.bm25).exists():
        raise FileNotFoundError(f"BM25 index file '{args.bm25}' not found.  Please run build_index.py first.")
    orc = Orchestrator(args.bm25, args.faiss)
    query = args.query
    for i in range(args.n_steps):
        result = orc.run_step(query)
        state = result.get("state")
        if result.get("refusal"):
            print(f"State {state}: {result['message']}")
            break
        print(f"\n=== State {state} ===")
        for item in result.get("checklist", []):
            print(f"- {item}")
        print("Citations:", "; ".join(result.get("citations", [])))
        if not result.get("state_complete", False):
            # For demo purposes, we consider the state complete when the model says so.
            # In a full implementation, you'd ask the user for more information before advancing.
            pass
        # Advance state for next iteration
        orc.sm.advance()
        if not orc.sm.has_more():
            break


if __name__ == "__main__":
    main()