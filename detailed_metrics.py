import streamlit as st
import json
import pandas as pd
import re
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from nltk.corpus import stopwords
import numpy as np
from collections import Counter

# Ensure required NLTK data is present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def run_detailed_metrics():
    st.title("Comprehensive Text Evaluation Metrics")

    # ---------- Utility Functions ----------
    def preprocess_text(text):
        if not isinstance(text, str):
            text = str(text)
        text = re.sub(r'\\[a-zA-Z]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def calculate_bleu(reference, candidate):
        reference_tokens = preprocess_text(reference).split()
        candidate_tokens = preprocess_text(candidate).split()
        if not reference_tokens or not candidate_tokens:
            return 0.0, 0.0, 0.0, 0.0
        smoothie = SmoothingFunction().method1
        return (
            sentence_bleu([reference_tokens], candidate_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie),
            sentence_bleu([reference_tokens], candidate_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie),
            sentence_bleu([reference_tokens], candidate_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie),
            sentence_bleu([reference_tokens], candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        )

    def calculate_rouge(reference, candidate):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(preprocess_text(reference), preprocess_text(candidate))
        return scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure

    def calculate_semantic_similarity(reference, candidate):
        try:
            texts = [preprocess_text(reference), preprocess_text(candidate)]
            if not all(texts): return 0.0
            tfidf = TfidfVectorizer(stop_words='english').fit_transform(texts)
            return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        except:
            return 0.0

    def calculate_exact_match(reference, candidate):
        return 1.0 if preprocess_text(reference).lower() == preprocess_text(candidate).lower() else 0.0

    def calculate_partial_match(reference, candidate):
        return SequenceMatcher(None, preprocess_text(reference), preprocess_text(candidate)).ratio()

    def calculate_keyword_overlap(reference, candidate):
        stop_words = set(stopwords.words('english'))
        ref_words = {w.lower() for w in preprocess_text(reference).split() if w.lower() not in stop_words}
        cand_words = {w.lower() for w in preprocess_text(candidate).split() if w.lower() not in stop_words}
        if not ref_words: return 0.0
        return len(ref_words & cand_words) / len(ref_words)

    def calculate_answer_length_ratio(reference, candidate):
        ref_len, cand_len = len(preprocess_text(reference).split()), len(preprocess_text(candidate).split())
        if ref_len == 0: return 1.0 if cand_len == 0 else 0.0
        return min(ref_len, cand_len) / max(ref_len, cand_len)

    def detect_answer_type(text):
        words = preprocess_text(text).split()
        word_count = len(words)
        if word_count <= 3:
            return "Binary" if text.strip().lower() in {"yes", "no", "true", "false"} else "Short"
        elif word_count < 10:
            return "Short"
        elif word_count < 50:
            return "Medium"
        else:
            return "Long"

    def calculate_comprehensive_scores(reference, candidate, ref_type, cand_type):
        bleu_1, bleu_2, bleu_3, bleu_4 = calculate_bleu(reference, candidate)
        rouge1, rouge2, rougeL = calculate_rouge(reference, candidate)

        return {
            "BLEU-1": bleu_1,
            # "BLEU-2": bleu_2,
            # "BLEU-3": bleu_3,
            # "BLEU-4": bleu_4,
            "ROUGE-1": rouge1,
            "ROUGE-2": rouge2,
            "ROUGE-L": rougeL,
            "Semantic Similarity": calculate_semantic_similarity(reference, candidate),
            "Exact Match": calculate_exact_match(reference, candidate),
            "Partial Match": calculate_partial_match(reference, candidate),
            "Keyword Overlap": calculate_keyword_overlap(reference, candidate),
            "Length Ratio": calculate_answer_length_ratio(reference, candidate),
            "Answer Type Match": 1.0 if ref_type == cand_type else 0.0,
            "Ref Type": ref_type,
            "Cand Type": cand_type
        }
    def extract_question_types(obj):
        """Recursively collect all question_type values in object"""
        types = []
        def recurse(o):
            if isinstance(o, dict):
                if 'type' in o:
                    types.append(o['type'])
                for v in o.values():
                    recurse(v)
            elif isinstance(o, list):
                for item in o:
                    recurse(item)
        recurse(obj)
        return types


    def extract_field_values(obj, field_name):
        values = []
        def recurse(o):
            if isinstance(o, dict):
                if field_name in o:
                    values.append(o[field_name])
                for v in o.values():
                    recurse(v)
            elif isinstance(o, list):
                for item in o:
                    recurse(item)
        recurse(obj)
        return values

    def extract_all_fields_recursive(obj):
        fields = set()
        def recurse(o):
            if isinstance(o, dict):
                fields.update(o.keys())
                for v in o.values():
                    recurse(v)
            elif isinstance(o, list):
                for item in o:
                    recurse(item)
        recurse(obj)
        return sorted(fields)

    # ---------- UI ----------
    col1, col2 = st.columns(2)
    with col1:
        golden_file = st.file_uploader("Upload Golden Data (JSON)", type="json", key="detailed_golden")
    with col2:
        prediction_file = st.file_uploader("Upload Model Predictions (JSON)", type="json", key="detailed_pred")

    if golden_file and prediction_file:
        try:
            golden_data = json.load(golden_file)
            prediction_data = json.load(prediction_file)

            fields_golden = extract_all_fields_recursive(golden_data)
            fields_pred = extract_all_fields_recursive(prediction_data)

            col1, col2 = st.columns(2)
            with col1:
                golden_field = st.selectbox("Select Field from Golden", fields_golden)
            with col2:
                prediction_field = st.selectbox("Select Field from Prediction", fields_pred)

            if st.button("Calculate Scores"):
                gold_texts = extract_field_values(golden_data, golden_field)
                pred_texts = extract_field_values(prediction_data, prediction_field)
                ref_types = extract_question_types(golden_data)
                cand_types = extract_question_types(prediction_data)
                min_len = min(len(gold_texts), len(pred_texts), len(ref_types), len(cand_types))

                st.write("🧪 Input length check:")
                st.write(f"gold_texts: {len(gold_texts)}")
                st.write(f"pred_texts: {len(pred_texts)}")
                st.write(f"ref_types: {len(ref_types)}")
                st.write(f"cand_types: {len(cand_types)}")
                results = []
                for i in range(min_len):
                    ref = gold_texts[i]
                    cand = pred_texts[i]
                    ref_type = ref_types[i] if i < len(ref_types) else "unknown"
                    cand_type = cand_types[i] if i < len(cand_types) else "unknown"
                    
                    # if not ref or not cand:
                    #     continue

                    row = calculate_comprehensive_scores(ref, cand, ref_type, cand_type)
                    row["Index"] = i
                    row["Reference"] = ref
                    row["Prediction"] = cand
                    results.append(row)


                if results:
                    df = pd.DataFrame(results)
                    st.subheader("Detailed Scores")
                    st.dataframe(df)


                    # Show average metrics
                    st.subheader("📈 Average Metrics")
                    avg_cols = [col for col in df.columns if df[col].dtype in [np.float64, float]]
                    avg_df = df[avg_cols].mean().reset_index()
                    avg_df.columns = ["Metric", "Average Score"]
                    st.dataframe(avg_df)

                    # 2️⃣ Line Chart – Metric Trends per Index
                    st.subheader("📉 Metric Trends by Index")
                    # Prepare data for line chart
                    metric_cols = ["BLEU-1", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Semantic Similarity", "Exact Match", "Partial Match", "Keyword Overlap", "Length Ratio", "Answer Type Match"]
                    score_melted = df.melt(id_vars=["Index"], value_vars=metric_cols, var_name="Metric", value_name="Score")
                    fig_line = px.line(score_melted, x="Index", y="Score", color="Metric", markers=True)
                    st.plotly_chart(fig_line, use_container_width=True)

                    # 📐 BLEU-1 by Answer Type (with labels)
                    st.subheader("📐 BLEU-1 Score by Answer Type (with labels)")
                    ref_bleu = df.groupby("Ref Type")["BLEU-1"].mean().reset_index()
                    ref_bleu["Source"] = "Reference"

                    cand_bleu = df.groupby("Cand Type")["BLEU-1"].mean().reset_index()
                    cand_bleu.columns = ["Type", "BLEU-1"]
                    cand_bleu["Source"] = "Prediction"

                    ref_bleu.columns = ["Type", "BLEU-1", "Source"]
                    bleu_by_type = pd.concat([ref_bleu, cand_bleu], ignore_index=True)

                    # 📊 Compare BLEU-1 vs ROUGE-1 vs Semantic Similarity by Ref Type
                    st.subheader("📊 BLEU-1 vs ROUGE-1 vs Semantic Similarity by Answer Type (Ref Type)")
                    agg_df = df.groupby("Ref Type")[["BLEU-1", "ROUGE-1", "Semantic Similarity"]].mean().reset_index()
                    metric_melted = agg_df.melt(id_vars="Ref Type", var_name="Metric", value_name="Score")

                    fig_comp = px.bar(metric_melted, x="Ref Type", y="Score", color="Metric", barmode="group", text="Score")
                    fig_comp.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                    fig_comp.update_layout(title="Metric Comparison by Answer Type", xaxis_title="Answer Type", yaxis_title="Score")
                    st.plotly_chart(fig_comp, use_container_width=True)

                    st.subheader("🔍 Per-Sample Comparison with Labels")

                    for i, row in df.iterrows():
                        with st.expander(f"Index {int(row['Index'])} — Ref Type: {row['Ref Type']} | Cand Type: {row['Cand Type']}", expanded=i == 0):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**🔹 Golden Text**")
                                st.text_area("", row["Reference"], height=150, key=f"ref_{i}")
                            with col2:
                                st.markdown("**🔸 Prediction Text**")
                                st.text_area("", row["Prediction"], height=150, key=f"pred_{i}")

                            st.markdown("**📊 Scores:**")
                            st.write({
                                "BLEU-1": round(row["BLEU-1"], 3),
                                "ROUGE-1": round(row["ROUGE-1"], 3),
                                "Semantic Similarity": round(row["Semantic Similarity"], 3),
                                "Exact Match": row["Exact Match"],
                                "Answer Type Match": row["Answer Type Match"]
                            })

                    st.download_button("Download Results", data=df.to_csv(index=False), file_name="detailed_metrics.csv")

        except Exception as e:
            st.error("Failed to process files.")
            st.exception(e)
    else:
        st.info("Please upload both files to continue.")
