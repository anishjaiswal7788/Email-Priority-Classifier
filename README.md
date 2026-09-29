# 📧 Automatic Email Priority Classifier

Classifies incoming emails into **High / Medium / Low** priority using classical NLP + machine learning, with a Streamlit demo that ranks a whole inbox.

## 1. Problem formulation & originality
**Problem:** Knowledge workers receive 100+ emails/day; important ones (deadlines, escalations, security alerts) get buried under newsletters and promos.
**Formulation:** 3-class supervised text classification, `f(subject, body, sender) -> {High, Medium, Low}`.
**What makes it different from plain spam detection:**
- Spam filtering is binary (junk vs. not). Priority is *ordinal* and *context-dependent*: a legitimate email can still be Low.
- Promotional emails deliberately imitate urgency ("URGENT!!! Last chance!") — a hard adversarial case we model explicitly.
- The **High-priority class matters most**: a missed urgent email costs far more than a mis-ranked newsletter, so we track High-class F1/recall separately and use `class_weight="balanced"`.
- Output is a **ranked inbox with confidence**, not just a label.

## 2. AI concepts implemented
Text preprocessing → **TF-IDF** (1–2 grams, sublinear tf) → classifiers → evaluation. Concepts: bag-of-words vs n-grams, sparse vectors, Naive Bayes (generative), Logistic Regression / Linear SVM (discriminative, max-margin), Decision Tree / Random Forest (non-linear, bagging), class imbalance, stratified k-fold CV, calibration via probabilities/softmax of margins.

## 3. Comparison of algorithms (5-fold CV + held-out 20% test)
See `results/comparison.csv` (reproduced by `python train.py`). Compared on accuracy, macro-F1, High-class F1, CV std (stability), training time, and inference latency.

| Algorithm | Why included |
|---|---|
| Multinomial Naive Bayes | fast probabilistic baseline for text |
| Logistic Regression | strong linear baseline, gives probabilities |
| Linear SVM | typically best on sparse high-dim text |
| Decision Tree | interpretable, non-linear |
| Random Forest | ensemble; tests whether non-linearity helps |

**Findings (synthetic data):** linear models (NB/LR/SVM) cluster at ~0.87 macro-F1, Random Forest ~0.83, single Decision Tree ~0.75–0.79 (overfits sparse features). Linear models win on both accuracy *and* speed, which matches theory: text is high-dimensional and near-linearly separable.

## 4. Dataset & experimentation
- `generate_data.py` builds 2,100 labelled emails (700/class, balanced) with realistic noise: 12% cross-class phrasing, "urgent-sounding" promos, and 7% label noise to mimic annotator disagreement.
- **Limitation (be upfront in viva):** synthetic data is templated, so absolute scores don't transfer to real inboxes. The pipeline accepts any CSV with `subject, body, sender, label`. **To strengthen the project, retrain on a real corpus** (e.g. Enron Email Dataset on Kaggle, or your own labelled emails) — this is the single biggest upgrade you can make.
- Split: 80/20 stratified, `random_state=42`; hyperparameters not tuned on test; model chosen by **CV score only**.

## 5. Evaluation & interpretation
Metrics: accuracy, precision/recall/F1 per class, macro-F1, confusion matrix (`results/confusion_matrix.png`), CV mean±std, error analysis (`results/errors.csv`). Macro-F1 is the headline metric because it weights all classes equally. Read the errors: most confusions are High↔Medium (adjacent priorities) and promos that shout urgency — errors are "near misses", not wild ones.

## 6. Own improvement / innovation
**Hybrid features:** TF-IDF + 8 engineered signals — urgency-word count, deadline regex (today/tomorrow/EOD), promo-word count, `!` count, ALL-CAPS ratio, VIP-sender flag, question mark, length. Ablation in `results/improvement.png`.
**Honest result:** on this synthetic data the engineered features did *not* significantly beat TF-IDF alone (differences are within CV std ≈ 0.02), because TF-IDF already captures the templated keywords and injected noise caps accuracy near ~87%. That is a legitimate finding: such features are expected to help more on real emails, where sender identity and formatting carry signal that words don't. Further ideas: sentence embeddings (SBERT), time-since-received, thread length, user feedback loop for online learning.

## 7. Demo
```bash
pip install -r requirements.txt
python generate_data.py && python train.py
python -m streamlit run app.py
```
Tab 1 classifies a single email with probability bars; Tab 2 uploads a CSV and returns it sorted High → Low.

## 8. Viva prep (short answers)
- **Why TF-IDF, not raw counts?** Down-weights common words, up-weights discriminative ones.
- **Why macro-F1?** Accuracy hides per-class failure; we care about High class.
- **Why SVM/LR beat Random Forest?** Sparse, high-dimensional text is close to linearly separable; trees split on one sparse feature at a time and overfit.
- **Why not select on the test set?** That leaks information; we pick by CV, test once.
- **Limitations?** Synthetic data, English only, no personalization, no attachments/threads.
- **Next step?** Real data, transformer embeddings, per-user feedback learning.

## Repo layout
`generate_data.py` · `features.py` · `train.py` · `app.py` · `data/` · `models/` · `results/`
