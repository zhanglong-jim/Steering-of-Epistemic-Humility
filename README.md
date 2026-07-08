# Minimal Anonymous Reproduction

This minimal package has only one code file and one instruction file.

- `verify_paper_results_one_cell.py`: copy-paste Colab cell that verifies the
  paper-facing results from the result zips.
- `README.md`: this file.

## How To Run

1. Open Colab.
2. Put these three result zips in `/content`, or keep them in a `results/`
   folder next to the code:
   - `ehc_v13_gpu_closure.zip`
   - `ehc_v18_gpu_oral_booster.zip`
   - `ehc_v21_judged_open_verifier.zip`
3. Copy the entire `verify_paper_results_one_cell.py` file into one Colab
   cell.
4. Click the Colab run button.

No command-line arguments, no `python script.py`, and no imports from local
project files are required.

## What It Verifies

The cell checks the manuscript's main reported results:

- forced-choice public operating points for Gemma 2B, Llama 3.1 8B, and Qwen3
  4B;
- six of six public curves reaching target intensity before high utility cost;
- six of six public curves having a low-cost useful operating point;
- SQuAD open-generation gains for Llama and Qwen under the stated cost budget;
- zero refusal-like outputs in the benign off-target audit;
- judged SQuAD open-generation results and the verifier comparison.

The expected final line is:

```text
FINAL VERDICT: PASS
```
