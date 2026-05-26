# 🔍 FakeShield — AI Misinformation Detection System

A complete, production-ready **Fake News Detection System** that fuses
deep contextual text understanding (BERT) with stylometric, temporal,
and heuristic **metadata signals**, then explains every decision using
**SHAP** and **LIME**.

---

## 📐 Architecture

```
                       ┌──────────────────────────┐
                       │   Raw Article (title,     │
                       │   text, subject, date)    │
                       └────────────┬──────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              │                                            │
   ┌──────────▼───────────┐                  ┌─────────────▼────────────┐
   │  BRANCH A: TEXT       │                  │  BRANCH B: METADATA      │
   │  combined_text →      │                  │  23 engineered features  │
   │  BERT [CLS] (768-d)   │                  │  (text/title/date/subj/  │
   │                       │                  │   credibility heuristic) │
   │  Linear 768→256       │                  │  Linear n→64             │
   │  BN→ReLU→Drop(0.3)    │                  │  BN→ReLU→Drop(0.2)       │
   │  Linear 256→128       │                  │  Linear 64→32            │
   │  ReLU→Drop(0.2)       │                  │  ReLU                    │
   └──────────┬───────────┘                  └─────────────┬────────────┘
              │  (128)                                      │  (32)
              └──────────────────┬──────────────────────────┘
                                 │  concat → (160)
                       ┌─────────▼──────────┐
                       │  FUSION HEAD       │
                       │  Linear 160→64     │
                       │  ReLU→Drop(0.2)    │
                       │  Linear 64→1       │
                       │  Sigmoid           │
                       └─────────┬──────────┘
                                 │
                  Misinformation Risk Score ∈ [0, 1]
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
        SHAP (metadata)    LIME (text)     Risk Tier + Prediction
```

---

## 📦 Dataset

Two CSV files placed in `data/`:

| File       | Rows    | Label |
|------------|---------|-------|
| `Fake.csv` | 23,502  | 1     |
| `True.csv` | 21,417  | 0     |

Shared columns: `title`, `text`, `subject`, `date`.
There is **no** pre-existing label column — labels are created during
preprocessing (`1 = Fake`, `0 = True`).

---

## 🗂️ Project Structure

```
fake_news_detector/
├── data/
│   ├── Fake.csv                (you provide)
│   ├── True.csv                (you provide)
│   └── cleaned_data.csv        (generated)
├── notebooks/
│   └── EDA.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── bert_embeddings.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explainability.py
│   └── predict.py
├── app/
│   └── streamlit_app.py
├── models/                     (generated weights/scalers/encoders)
├── plots/                      (generated figures)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> First run downloads `bert-base-uncased` (~420 MB) from HuggingFace
> and caches it locally. Subsequent runs are offline.

---

## 🚀 How to Run (data → training → app)

```bash
# 1. Place Fake.csv and True.csv in data/

# 2. Clean & merge the data
python src/data_preprocessing.py

# 3. (Optional) Explore the data
jupyter notebook notebooks/EDA.ipynb

# 4. Train the multi-input model (caches BERT embeddings)
python src/train.py

# 5. Evaluate (plots + comparison table)
python src/evaluate.py

# 6. Generate SHAP / LIME explainability artifacts
python src/explainability.py

# 7. Predict a single article from the CLI
python src/predict.py

# 8. Launch the web app
streamlit run app/streamlit_app.py
```

---

## 🚦 Risk Tiers

| Score range  | Tier        | Meaning              |
|--------------|-------------|----------------------|
| 0.00 – 0.30  | ✅ LOW       | Likely Real          |
| 0.30 – 0.60  | ⚠️ MEDIUM    | Uncertain            |
| 0.60 – 0.85  | 🔶 HIGH      | Likely Fake          |
| 0.85 – 1.00  | 🚨 CRITICAL  | Very Likely Fake     |

Decision threshold: **score ≥ 0.5 → FAKE**, otherwise **REAL**.

---

## 📈 Model Performance (placeholder — fill after `evaluate.py`)

| Model                          | Accuracy | F1 (Fake) | ROC-AUC |
|--------------------------------|----------|-----------|---------|
| Multi-Input (BERT + Metadata)  |   _TBD_  |   _TBD_   |  _TBD_  |
| Baseline LogReg (Metadata only)|   _TBD_  |   _TBD_   |  _TBD_  |

Generated automatically into `plots/model_comparison.csv`.

---

## 🖼️ Sample Output Screenshots (placeholder)

| Artifact                         | Location                              |
|----------------------------------|---------------------------------------|
| Training curves                  | `plots/training_curves.png`           |
| Confusion matrices               | `plots/confusion_matrix.png`          |
| ROC / PR curves                  | `plots/roc_pr_curves.png`             |
| Risk-score distribution          | `plots/risk_score_distribution.png`   |
| SHAP summary / beeswarm          | `plots/shap_*.png`                    |
| LIME example explanation         | `plots/lime_example_explanation.html` |

---

## 🔮 Future Improvements

- Fine-tune BERT end-to-end instead of using frozen `[CLS]` features.
- Add source/domain reputation features and fact-check API signals.
- Multilingual support (XLM-RoBERTa).
- Calibrated probabilities (temperature / isotonic scaling).
- Active-learning loop for human-in-the-loop label correction.
- Dockerized deployment + REST API.

---

## ⚠️ Disclaimer

This tool is for **educational purposes**. Always verify news with
trusted sources. Model predictions are probabilistic and can be wrong.

---

## 📄 License

Released under the **MIT License**. You are free to use, modify, and
distribute this project with attribution. See `LICENSE` for details.
