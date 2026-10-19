# Kino: notebooks to run from your account (20 Oct)

Our GPU quota is used up until 24 Oct, 5:30 am IST, so these need your quota. Tell Aryan which ones you submitted, so nothing is submitted twice.

## 1. CASMI26: the public 0.456 + depth 7 (one find-and-replace)
1. Open https://www.kaggle.com/code/guhongbin/casmi26-v465-v15-pop015 and click **Copy & Edit**.
2. Find `max_depth=6, max_iter=500` and replace it with `max_depth=7, max_iter=500`. There are exactly **2** hits, both inside the embedded engine file text; change both. Change nothing else.
3. In the right-hand panel set Accelerator **GPU T4 ×1**, Internet **off**, Environment **Pin to original**.
4. Click **Save Version → Save & Run All**. When the run finishes, submit `submission.csv`.

Forecast 0.456–0.458. Depth 7 gave +0.003 and +0.004 on two earlier bases and tied on a third. Run-to-run noise is about ±0.004.

## 2. NTL: the 0.594 + our retrained 8B adapter (not ready yet)
Low priority: the 0.592 author found that seed averaging didn't help (forecast 0.594, likely a tie). It waits until the Colab training finishes (about 8 pm IST tonight). Aryan will send the notebook file and the dataset link if we go ahead.
