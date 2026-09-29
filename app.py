import joblib, numpy as np, pandas as pd, streamlit as st
from features import prepare, meta_features   # needed for unpickling
st.set_page_config(page_title="Email Priority Classifier", page_icon="📧")
st.title("📧 Automatic Email Priority Classifier")
model = joblib.load("models/best_model.joblib")
COLOR = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}
def score(df):
    d = prepare(df); pred = model.predict(d)
    if hasattr(model, "predict_proba"): pr = model.predict_proba(d)
    else:
        z = model.decision_function(d); e = np.exp(z - z.max(1, keepdims=True)); pr = e / e.sum(1, keepdims=True)
    return pred, pr, model.classes_
tab1, tab2 = st.tabs(["Single email", "Inbox (CSV)"])
with tab1:
    s = st.text_input("Sender", "client.rep@acme.com"); sub = st.text_input("Subject", "Contract renewal at risk")
    b = st.text_area("Body", "Please call me immediately, we need to resolve this today!")
    if st.button("Classify"):
        pred, pr, cl = score(pd.DataFrame([{"subject": sub, "body": b, "sender": s}]))
        st.subheader(f"{COLOR[pred[0]]} {pred[0]} priority")
        st.bar_chart(pd.Series(pr[0], index=cl))
with tab2:
    f = st.file_uploader("CSV with columns: subject, body, sender", type="csv")
    if f:
        d = pd.read_csv(f)
        st.caption(f"{len(d):,} rows loaded")
        if len(d) > 5000:
            st.warning(f"Large file ({len(d):,} rows). Scoring in batches and showing the top 2,000 by priority — "
                       "full results are still computed and available to download below.")
        BATCH = 5000
        preds, confs = [], []
        prog = st.progress(0.0) if len(d) > BATCH else None
        for i in range(0, len(d), BATCH):
            chunk = d.iloc[i:i + BATCH]
            p, pr, cl = score(chunk)
            preds.append(p); confs.append(pr.max(1))
            if prog: prog.progress(min(1.0, (i + BATCH) / len(d)))
        d["priority"] = np.concatenate(preds)
        d["confidence"] = np.round(np.concatenate(confs), 2)
        d["rank"] = d.priority.map({"High": 0, "Medium": 1, "Low": 2})
        d = d.sort_values(["rank", "confidence"], ascending=[True, False]).drop(columns="rank")
        st.dataframe(d.head(2000) if len(d) > 2000 else d)
        st.download_button("Download full sorted results (CSV)", d.to_csv(index=False), "sorted_inbox.csv", "text/csv")
