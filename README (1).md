# ⚾ MLB Award Predictor

A Streamlit app that predicts MLB award vote share (MVP, Cy Young, Rookie of the Year) from player statistics and media coverage text, using a two-stage LightGBM model with Sentence-BERT NLP features.

---

## How It Works

The model uses two inputs to generate predictions:

**1. Player Statistics**
Batting (AVG, OPS, HR, RBI, SB, WAR), pitching (ERA, WHIP, W, SO, IP, K/BB), and fielding (Fielding %, DRS) statistics for the hypothetical season.

**2. Media Coverage (NLP)**
Optional — paste any article or narrative text about the player's season. The app embeds it using [Sentence-BERT (`all-MiniLM-L6-v2`)](https://www.sbert.net/) and extracts VADER sentiment, then feeds those features into the model alongside the stats.

**Model architecture:**
- Stage 1 (LightGBM Classifier): Predicts probability of receiving any award votes
- Stage 2 (LightGBM Regressor): Predicts vote share among players who received votes
- Final score: Stage1_prob × Stage2_share
- Tuned with Optuna (40 trials per stage per award)
- Trained on: 1992–2021 | Test set: 2022+

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/mlb-award-predictor.git
cd mlb-award-predictor
```

### 2. Add the model file

The trained model (`mlb_award_model.pkl`) is **not included** in this repo due to file size. Generate it from Google Colab using `mlb_nlp_sbert.ipynb` and place it in the repo root:

```
mlb-award-predictor/
├── app.py
├── requirements.txt
├── mlb_award_model.pkl   ← add this
└── README.md
```

To generate the pkl from Colab, add this cell at the end of the notebook:

```python
import pickle, os
from google.colab import files

model_bundle = {
    'results': {
        award: {
            'clf':      results[award]['clf'],
            'reg':      results[award]['reg'],
            'prep':     results[award]['prep'],
            'features': results[award]['features'],
        }
        for award in results
    },
    'pca':              pca,
    'sbert_model_name': 'all-MiniLM-L6-v2',
    'NLP_FEATURE_COLS': NLP_FEATURE_COLS,
    'FEATURE_COLS_BASE': FEATURE_COLS_BASE,
    'FEATURE_COLS_NLP':  FEATURE_COLS_NLP,
}

with open('/content/mlb_award_model.pkl', 'wb') as f:
    pickle.dump(model_bundle, f)

print(f"Size: {os.path.getsize('/content/mlb_award_model.pkl')/1e6:.1f} MB")
files.download('/content/mlb_award_model.pkl')
```

> **⚠️ Large file warning:** If `mlb_award_model.pkl` exceeds ~100MB, use [Git LFS](https://git-lfs.com/) or host the file externally (Google Drive, Hugging Face Hub) and load it at runtime.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run locally

```bash
streamlit run app.py
```

---

## Deploying to Streamlit Cloud

1. Push this repo to GitHub (without the `.pkl` — see note above)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub repo
3. Set **Main file path** to `app.py`
4. Before deploying, upload `mlb_award_model.pkl` — options:
   - **Git LFS** (recommended if file < 2GB): `git lfs track "*.pkl"` then commit normally
   - **Manual upload**: Streamlit Cloud allows file uploads via their dashboard under Advanced Settings

> Streamlit Cloud has a **1GB slug limit**. If your pkl is large, consider trimming the bundle (remove raw dataframes, keep only clf/reg/prep/pca objects).

---

## Repo Structure

```
mlb-award-predictor/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── .gitignore
├── README.md
└── mlb_award_model.pkl     # (not in repo — add locally or via LFS)
```

---

## Model Performance

| Award    | Features  | Test Spearman | Test PR-AUC |
|----------|-----------|---------------|-------------|
| MVP      | Base      | —             | —           |
| MVP      | Base+NLP  | —             | —           |
| Cy Young | Base      | —             | —           |
| Cy Young | Base+NLP  | —             | —           |

*(Fill in from your Colab output after training)*

---

## Notes & Limitations

- Predictions reflect historical voting patterns from 1992–2021. Voter preferences shift over time.
- The NLP component expects descriptive narrative text — short inputs or non-English text will reduce prediction quality.
- ROY predictions require the player to be a genuine rookie; the model has no way to verify rookie eligibility from stats alone.
- The model was trained on a specific tabular feature schema. Mismatched column names will result in NaN-imputed features, degrading accuracy.
