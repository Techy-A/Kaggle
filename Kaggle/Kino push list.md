# Kino: notebooks to run from your account (18 Oct, only if ours are still queued at 10 pm IST)

Our account's GPU runs have been stuck in Kaggle's queue since about 5:30 pm IST, and we have only 3.1 GPU hours left this week. These are public notebooks, so you can fork them yourself. Each one is a single team submission. Please submit before the **5:30 am IST** reset, in this order (highest value first). Tell Aryan which ones you submitted, so ours don't get submitted twice.

For every notebook: open the link, **Copy & Edit**, then in the right-hand panel:
- Accelerator: **GPU T4 ×1**
- Internet: **off**
- Environment: set as noted below

Then **Save Version → Save & Run All**. When the run finishes, open the version and click **Submit** (CASMI file `submission.csv`, NTL file `submission.parquet`).

| # | Competition | Notebook | Environment | Why | Forecast |
|---|---|---|---|---|---|
| 1 | Name That LLM | https://www.kaggle.com/code/obstacledeveloper/local-candidate-adaptive-centered-stylegram | **Pin to original** | exact copy of the public 0.559 | 0.559 |
| 2 | CASMI26 | https://www.kaggle.com/code/ahmedberatozer/casmi26-v14-pubchemlite | **Always use latest environment** | public 0.455 code on the newer image (the image gave v459 +0.008) | 0.459 |
| 3 | CASMI26 | https://www.kaggle.com/code/ahmedberatozer/casmi26-v14-pubchemlite | **Pin to original** | exact copy of the public 0.455 | 0.455 |
| 4 | CASMI26 | https://www.kaggle.com/code/seyitkaangunes/casmi26-e006-n1a-cc-by-chebi-lb-0-452 | **Always use latest environment** | public 0.452 code on the newer image | 0.457 |

NTL has only **1** slot left today; CASMI has **4**.

If "latest environment" isn't offered, check the notebook's version settings: the newer image is the one guhongbin's v459 "defaultimg" uses.
