"""Compare 5 classifiers on TF-IDF (baseline) vs TF-IDF + engineered features (our improvement)."""
import time, joblib, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, ConfusionMatrixDisplay
from features import prepare, meta_features

df = prepare(pd.read_csv("data/emails.csv"))
X, y = df[["subject", "body", "sender", "text"]], df["label"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
ORDER = ["High", "Medium", "Low"]

def build(clf, improved):
    tf = ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), "text")
    parts = [tf]
    if improved:
        parts.append(("meta", Pipeline([("f", FunctionTransformer(meta_features)), ("s", MinMaxScaler())]), ["subject", "body", "sender"]))
    return Pipeline([("ct", ColumnTransformer(parts)), ("clf", clf)])

MODELS = {
 "Naive Bayes": lambda: MultinomialNB(),
 "Logistic Regression": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
 "Linear SVM": lambda: LinearSVC(class_weight="balanced"),
 "Decision Tree": lambda: DecisionTreeClassifier(max_depth=20, random_state=0),
 "Random Forest": lambda: RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1),
}
cv = StratifiedKFold(5, shuffle=True, random_state=42)
rows, fitted = [], {}
for improved in (False, True):
    for name, mk in MODELS.items():
        if improved and name == "Naive Bayes": continue   # NB cannot use scaled non-count features fairly; skip
        pipe = build(mk(), improved)
        cvs = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="f1_macro", n_jobs=1)
        t = time.time(); pipe.fit(Xtr, ytr); ft = time.time() - t
        t = time.time(); p = pipe.predict(Xte); pt = (time.time() - t) / len(Xte) * 1000
        rows.append(dict(model=name, features="TF-IDF + engineered" if improved else "TF-IDF only",
                         cv_f1_mean=cvs.mean(), cv_f1_std=cvs.std(), test_acc=accuracy_score(yte, p),
                         test_f1_macro=f1_score(yte, p, average="macro"),
                         f1_high=f1_score(yte, p, labels=["High"], average="macro"),
                         train_s=ft, ms_per_email=pt))
        fitted[(name, improved)] = (pipe, p)
res = pd.DataFrame(rows).round(4).sort_values("test_f1_macro", ascending=False)
res.to_csv("results/comparison.csv", index=False); print(res.to_string(index=False))

best = res.sort_values("cv_f1_mean", ascending=False).iloc[0]  # select on CV only, never on test
key = (best.model, best.features != "TF-IDF only"); pipe, p = fitted[key]
print("\nBEST:", key); print(classification_report(yte, p, labels=ORDER))
joblib.dump(pipe, "models/best_model.joblib")
ConfusionMatrixDisplay(confusion_matrix(yte, p, labels=ORDER), display_labels=ORDER).plot(cmap="Blues")
plt.title(f"Confusion matrix: {key[0]}"); plt.savefig("results/confusion_matrix.png", dpi=150, bbox_inches="tight"); plt.close()

# ablation chart: baseline vs improved
piv = res.pivot(index="model", columns="features", values="test_f1_macro").dropna()
piv.plot.bar(rot=20); plt.ylabel("Test macro-F1"); plt.title("Effect of engineered features"); plt.ylim(0.6, 1)
plt.savefig("results/improvement.png", dpi=150, bbox_inches="tight")

# error analysis
err = Xte.assign(true=yte, pred=p)[yte != p]; err.head(25).to_csv("results/errors.csv", index=False)
print(f"\n{len(err)} misclassified of {len(yte)}; saved sample to results/errors.csv")
