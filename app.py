import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import spacy
import textstat
import nltk
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

# ───────────────────────────────── 0) Downloads
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("vader_lexicon")
nltk.download("averaged_perceptron_tagger")

# ───────────────────────────────── 1) Globals
nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))
CLICKBAIT = {
    "shocking",
    "you",
    "won't believe",
    "top",
    "never",
    "amazing",
    "secret",
    "insane",
}
PIPELINE_FILE = "pipeline.joblib"


# ───────────────────────────────── 2) Cleaner
def clean(text: str) -> str:
    lower = text.lower()
    tokens = [t for t in word_tokenize(lower) if t.isalpha() and t not in stop_words]
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(lemmas)


# ───────────────────────────────── 3) Plagiarism detector
class PlagiarismDetector:
    def __init__(self, t_csv, f_csv, cache="plagiarism_cache", k=5, thr=0.70):
        self.thr, self.k = thr, k
        os.makedirs(cache, exist_ok=True)
        v_path = f"{cache}/vec.joblib"
        n_path = f"{cache}/nn.joblib"
        m_path = f"{cache}/meta.parquet"
        if all(map(os.path.exists, [v_path, n_path, m_path])):
            self.vec = joblib.load(v_path)
            self.nn = joblib.load(n_path)
            self.meta = pd.read_parquet(m_path)
        else:
            df = pd.concat(
                [
                    pd.read_csv(t_csv).assign(label=1),
                    pd.read_csv(f_csv).assign(label=0),
                ],
                ignore_index=True,
            )
            df["clean"] = df["text"].astype(str).apply(clean)
            self.vec = TfidfVectorizer(
                max_features=5000, ngram_range=(1, 2), tokenizer=str.split
            )
            X = self.vec.fit_transform(df["clean"])
            self.nn = NearestNeighbors(metric="cosine", n_neighbors=k).fit(X)
            self.meta = df[["title", "label", "clean"]]
            joblib.dump(self.vec, v_path)
            joblib.dump(self.nn, n_path)
            self.meta.to_parquet(m_path)

    def query(self, text: str):
        v = self.vec.transform([clean(text)])
        dist, idx = self.nn.kneighbors(v, self.k)
        sims = 1 - dist[0]
        rows = []
        for r, (i, s) in enumerate(zip(idx[0], sims), 1):
            rec = self.meta.iloc[i]
            rows.append(
                {
                    "rank": r,
                    "sim": float(s),
                    "title": str(rec.title)[:120],
                    "label": "True" if rec.label else "Fake",
                    "snippet": rec.clean[:120] + "…",
                }
            )
        return sims[0] >= self.thr, rows


detector = PlagiarismDetector("dataset/True.csv", "dataset/Fake.csv")


#───────────────────────────────── 4) Data loader
@st.cache_data
def load_data(t_path,f_path):
    t=pd.read_csv(t_path);f=pd.read_csv(f_path)
    t["label"]=1;f["label"]=0;return pd.concat([t,f],ignore_index=True)


#───────────────────────────────── 5) Feature extractor
def preprocess_and_extract_features(texts):
    cl,fe=[],[]
    for d in texts:
        low=d.lower(); fles=textstat.flesch_reading_ease(low); gun=textstat.gunning_fog(low)
        toks=[t for t in word_tokenize(low) if t.isalpha() and t not in stop_words]
        lem=[lemmatizer.lemmatize(t) for t in toks]; c=" ".join(lem); cl.append(c)
        tags=pos_tag(lem); tot=len(tags)or 1
        adj=sum(p.startswith("JJ") for _,p in tags)/tot; noun=sum(p.startswith("NN") for _,p in tags)/tot
        ents=[e.label_ for e in nlp(c).ents]; per,org,gpe=ents.count("PERSON"),ents.count("ORG"),ents.count("GPE")
        vs=sia.polarity_scores(low); cb=sum(low.count(w) for w in CLICKBAIT)
        fe.append([fles,gun,adj,noun,per,org,gpe,vs["compound"],vs["pos"],vs["neg"],cb])
    return cl,np.array(fe)


#───────────────────────────────── 6) Topic modeling
def get_topic_distributions(texts: list[str], n_topics: int = 5):
    cv = CountVectorizer(max_features=5000, stop_words="english")
    counts = cv.fit_transform(texts)
    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=42, learning_method="batch"
    )
    tops = lda.fit_transform(counts)
    return cv, lda, tops


#───────────────────────────────── 6) Full pipeline
@st.cache_resource
def run_pipeline(data: pd.DataFrame) -> dict:
    # Demo sample
    sample = data["text"].iloc[0]
    demo_clean, demo_feats = preprocess_and_extract_features([sample])

    # Full texts
    texts = data["text"].tolist()
    clean_texts, dense_feats = preprocess_and_extract_features(texts)

    # Baseline TF-IDF
    tfidf_base = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_base = tfidf_base.fit_transform(clean_texts)

    # Enriched: char TF-IDF
    tfidf_char = TfidfVectorizer(max_features=2000, analyzer="char", ngram_range=(3, 5))
    X_char = tfidf_char.fit_transform(clean_texts)

    # Topics
    cv_count, lda_model, topic_feats = get_topic_distributions(clean_texts)

    # Combine enriched features
    X_full = hstack([X_base, X_char, dense_feats, topic_feats])

    y = data["label"].values
    # stratified split
    n_cls, n_samp = len(np.unique(y)), len(y)
    test_n = max(int(n_samp * 0.2), n_cls)
    Xtr_b, Xte_b, ytr, yte = train_test_split(
        X_base, y, test_size=test_n, random_state=42, stratify=y
    )
    Xtr_f, Xte_f, _, _ = train_test_split(
        X_full, y, test_size=test_n, random_state=42, stratify=y
    )

    # Train baseline models
    dt_base = DecisionTreeClassifier().fit(Xtr_b, ytr)
    lr_base = LogisticRegression(max_iter=1000).fit(Xtr_b, ytr)
    # Train enriched models
    dt_full = DecisionTreeClassifier().fit(Xtr_f, ytr)
    lr_full = LogisticRegression(max_iter=1000).fit(Xtr_f, ytr)

    # Predictions
    pred_dt_b = dt_base.predict(Xte_b)
    pred_lr_b = lr_base.predict(Xte_b)
    pred_dt_f = dt_full.predict(Xte_f)
    pred_lr_f = lr_full.predict(Xte_f)

    # Confusion matrices
    cms = {
        "DT Baseline": confusion_matrix(yte, pred_dt_b),
        "LR Baseline": confusion_matrix(yte, pred_lr_b),
        "DT Enriched": confusion_matrix(yte, pred_dt_f),
        "LR Enriched": confusion_matrix(yte, pred_lr_f),
    }

    # Metrics
    metrics = {}
    for name, preds in [
        ("DT Baseline", pred_dt_b),
        ("LR Baseline", pred_lr_b),
        ("DT Enriched", pred_dt_f),
        ("LR Enriched", pred_lr_f),
    ]:
        metrics[name] = {
            "Accuracy": accuracy_score(yte, preds),
            "Precision": precision_score(yte, preds, zero_division=0),
            "Recall": recall_score(yte, preds, zero_division=0),
            "F1": f1_score(yte, preds, zero_division=0),
        }

    return {
        # demo
        "sample_text": sample,
        "demo_clean": demo_clean[0],
        "demo_feats": demo_feats[0],
        # shapes & size
        "n_docs": len(texts),
        "Xb_shape": X_base.shape,
        "Xf_shape": X_full.shape,
        # models & artifacts
        "tfidf_base": tfidf_base,
        "tfidf_char": tfidf_char,
        "cv_count": cv_count,
        "lda": lda_model,
        "dt_base": dt_base,
        "lr_base": lr_base,
        "dt_full": dt_full,
        "lr_full": lr_full,
        # evaluation
        "metrics_df": pd.DataFrame(metrics).T,
        "conf_matrices": cms,
    }


#───────────────────────────────── 7) Check if the pipeline is loaded
def load_or_train_pipeline(data: pd.DataFrame) -> dict:
    if os.path.exists(PIPELINE_FILE):
        return joblib.load(PIPELINE_FILE)
    pipe = run_pipeline(data)
    joblib.dump(pipe, PIPELINE_FILE)
    return pipe


#───────────────────────────────── 8) NLP demo steps + explanations
def demo_nlp_steps(text: str) -> dict:
    lower = text.lower()
    tokens = word_tokenize(lower)
    tokens_ns = [t for t in tokens if t.isalpha() and t not in stop_words]
    lemmas = [lemmatizer.lemmatize(t) for t in tokens_ns]
    feats = preprocess_and_extract_features([text])[1][0]
    names = [
        "Flesch Reading Ease",
        "Gunning Fog Index",
        "Adjective Ratio",
        "Noun Ratio",
        "PERSON Count",
        "ORG Count",
        "GPE Count",
        "Sentiment Compound",
        "Sentiment Positive",
        "Sentiment Negative",
        "Clickbait Count",
    ]
    descs = {
        names[i]: d
        for i, d in enumerate(
            [
                "Higher → easier to read",
                "Higher → more complex",
                "Adjective tokens / total",
                "Noun tokens / total",
                "SpaCy PERSON entities",
                "SpaCy ORG entities",
                "SpaCy GPE entities",
                "VADER compound score [-1..1]",
                "VADER positive score",
                "VADER negative score",
                "Count of clickbait words",
            ]
        )
    }
    feat_dict = {names[i]: feats[i] for i in range(len(names))}
    return {
        "lower": lower,
        "tokens": tokens,
        "tokens_nostop": tokens_ns,
        "lemmas": lemmas,
        "features": feat_dict,
        "descriptions": descs,
    }


#───────────────────────────────── 9) Streamlit multipage
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    [
        "1. Train & Evaluate Models",
        "2. Pre-processing Walk-Through",
        "3. Real-Time Fake-News Prediction",
        "4. Plagiarism Similarity Search",
    ],
)

data = load_data(
    "C:/Users/Asus/Downloads/News Classifier/dataset/True.csv",
    "C:/Users/Asus/Downloads/News Classifier/dataset/Fake.csv",
)

if page == "1. Train & Evaluate Models":
    st.title("1️⃣ Train, Evaluate & Visualise Models")
    if st.button("▶️ Start training the model (or load it)"):
        pipe = load_or_train_pipeline(data)
        st.session_state["pipe"] = pipe
        st.session_state["ran"] = True
        st.success("✅ Pipeline ready!")
    if st.session_state.get("ran"):
        pipe = st.session_state["pipe"]

        st.subheader("Dataset Size & Feature Shapes")
        c1, c2 = st.columns(2)
        c1.metric("Documents", pipe["n_docs"])
        c2.metric(
            "Baseline TF-IDF shape", f"{pipe['Xb_shape'][0]}×{pipe['Xb_shape'][1]}"
        )
        st.metric(
            "Enriched feature matrix shape",
            f"{pipe['Xf_shape'][0]}×{pipe['Xf_shape'][1]}",
        )

        st.subheader("Performance Metrics")
        st.dataframe(pipe["metrics_df"], use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        pipe["metrics_df"].plot(kind="bar", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        st.pyplot(fig)

        st.subheader("Confusion Matrices")
        cms = pipe["conf_matrices"]
        for name, cmap in [
            ("DT Baseline", "Blues"),
            ("LR Baseline", "Purples"),
            ("DT Enriched", "Greens"),
            ("LR Enriched", "Oranges"),
        ]:
            st.markdown(f"**{name}**")
            fig, ax = plt.subplots()
            sns.heatmap(
                cms[name],
                annot=True,
                fmt="d",
                cmap=cmap,
                xticklabels=["Fake", "True"],
                yticklabels=["Fake", "True"],
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

elif page == "2. Pre-processing Walk-Through":
    st.title("2️⃣ NLP Step-by-Step Demo")
    if not st.session_state.get("ran"):
        st.warning("▶️ First run models on Page 1")
    else:
        pipe = st.session_state["pipe"]
        steps = demo_nlp_steps(pipe["sample_text"])
        st.subheader("Original Sample Text")
        st.write(pipe["sample_text"][:200] + "…")
        st.subheader("Transformations")
        st.write("**Lowercased:**", steps["lower"][:200] + "…")
        st.write("**Tokenized:**", steps["tokens"][:20])
        st.write("**No stopwords:**", steps["tokens_nostop"][:20])
        st.write("**Lemmatized:**", steps["lemmas"][:20])

        st.subheader("Dense Features Explanation")
        df_feats = pd.DataFrame.from_dict(
            steps["features"], orient="index", columns=["Value"]
        )
        df_feats["Description"] = df_feats.index.map(steps["descriptions"])
        st.table(df_feats)

elif page == "3. Real-Time Fake-News Prediction":
    st.title("3️⃣ Classify Custom Text in Real Time")
    if not st.session_state.get("ran"):
        st.warning("▶️ First run models on Page 1")
    else:
        pipe = st.session_state["pipe"]
        user_input = st.text_area("Enter a sentence:")
        if st.button("🔍 Classify"):
            steps = demo_nlp_steps(user_input)
            st.subheader("NLP Steps on Input")
            st.write("**Lowercased:**", steps["lower"])
            st.write("**Tokenized:**", steps["tokens"])
            st.write("**No stopwords:**", steps["tokens_nostop"])
            st.write("**Lemmatized:**", steps["lemmas"])
            df_in = pd.DataFrame.from_dict(
                steps["features"], orient="index", columns=["Value"]
            )
            df_in["Description"] = df_in.index.map(steps["descriptions"])
            st.table(df_in)

            # Baseline prediction
            clean_in, feat_dummy = steps["lower"], None
            Xb_in = pipe["tfidf_base"].transform([steps["lower"]])
            # Enriched prediction
            Xc_in = pipe["tfidf_char"].transform([steps["lower"]])
            cv_in = pipe["cv_count"].transform([steps["lower"]])
            top_in = pipe["lda"].transform(cv_in)
            feat_arr = np.array([list(steps["features"].values())])
            Xf_in = hstack([Xb_in, Xc_in, feat_arr, top_in])

            st.subheader("Predictions")
            for name, model, X_in in [
                ("DT Baseline", pipe["dt_base"], Xb_in),
                ("LR Baseline", pipe["lr_base"], Xb_in),
                ("DT Enriched", pipe["dt_full"], Xf_in),
                ("LR Enriched", pipe["lr_full"], Xf_in),
            ]:
                pred = model.predict(X_in)[0]
                proba = model.predict_proba(X_in)[0][pred]
                label = "True" if pred == 1 else "Fake"
                st.write(f"**{name}:** {label} _(prob={proba:.2f})_")

elif page == "4. Plagiarism Similarity Search":
    st.title("4️⃣ Plagiarism / Similarity Detection")
    txt = st.text_area("Paste a news article or paragraph:")
    if st.button("🔎 Find Similar Articles") and txt:
        plag, rows = detector.query(txt)
        st.subheader("Verdict")
        st.error(
            "⚠️ Potential plagiarism detected!" if plag else "✅ No close match found."
        )
        st.subheader("Top similar corpus articles")
        for r in rows:
            st.write(f"**{r['rank']}** (sim={r['sim']:.3f}) — {r['label']} news")
            st.write(f"*{r['title']}*")
            st.caption(r["snippet"])
