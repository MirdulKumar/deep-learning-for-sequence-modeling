# DNN Assignment 2 — RNN, LSTM, GRU and Transformer

**Student:** __________  
**Student ID:** __________  
**Variant (declared):** hidden size h=64, Transformer d=128, heads=4, layers=4, seeds {0,1,2}  
**Ablation parity:** last digit of student ID = _ (even → gradient clipping; odd → LSTM forget-bias)

## Environment
- Python 3.10, PyTorch 2.x, NumPy, pandas, Matplotlib, scikit-learn (see `requirements.txt`)
- Runs on Colab T4 GPU (`torch.cuda.is_available()` must return `True`)
- Total run time on T4: approx. 4–5 hours (Problem 1 ~3–4 h, Problem 2 ~1–2 h)

## How to run
1. Open `problem1/problem1.ipynb` in Colab, set runtime to T4 GPU, run `Restart & Run All`.
2. Open `problem2/problem2.ipynb`, same.
3. Both notebooks use relative paths only and save `results.csv` after every run.

## Results
- `results.csv`: one row per run (12 fixed columns).
- `problem1/figures/`: class balance, confusion matrix, adding curves, gradient norm, hidden-size sweep, ablation.
- `problem2/figures/`: attention heatmaps, learning curves, heads ablation, no-mask, time/memory vs context.
- `problem2/generated_samples.txt`: 500-char samples at temperature 0.2 / 0.8 / 1.5.

## Notes / assumptions
- The "Individualisation" variants table was not in the assignment document; defaults above are used and declared here.
- UCI HAR inertial signals are pre-normalised; recomputed per-channel stats ≈ 0/1 (expected).
- Problem 2 notebook builds attention / multi-head / encoder block from scratch (no `nn.MultiheadAttention` / `nn.Transformer*`).

## Anything that does not work
- None known at submission time.
