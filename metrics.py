import streamlit as st
import json
import pandas as pd
import re
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import plotly.express as px
import batch_metrics
import detailed_metrics

# Download NLTK tokenizer if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set page config
st.set_page_config(page_title="Text Comparison Scores", layout="wide")
st.title("ROUGE and BLEU Score Calculator")

# Tabs for two modes
tab1, tab2, tab3 = st.tabs(["Single File Comparison", "Batch File Evaluation","detailed metrics evaluation"])

# Shared utility functions
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_bleu(reference, candidate):
    ref_processed = preprocess_text(reference)
    cand_processed = preprocess_text(candidate)
    reference_tokens = ref_processed.split()
    candidate_tokens = cand_processed.split()
    if not reference_tokens or not candidate_tokens:
        return 0.0, 0.0, 0.0, 0.0
    smoothie = SmoothingFunction().method1
    bleu_1 = sentence_bleu([reference_tokens], candidate_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu_2 = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu_3 = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
    bleu_4 = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    return bleu_1, bleu_2, bleu_3, bleu_4

def calculate_rouge(reference, candidate):
    ref_processed = preprocess_text(reference)
    cand_processed = preprocess_text(candidate)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ref_processed, cand_processed)
    return scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure

def extract_all_fields_recursive(obj, field_set=None):
    if field_set is None:
        field_set = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            field_set.add(key)
            extract_all_fields_recursive(value, field_set)
    elif isinstance(obj, list):
        for item in obj:
            extract_all_fields_recursive(item, field_set)
    return sorted(field_set)

# ----------------------- TAB 1: Single File Evaluation -----------------------
with tab1:
    st.header("Upload Files")
    col1, col2 = st.columns(2)
    with col1:
        golden_file = st.file_uploader("Upload Golden Data (JSON)", type="json", key="golden")
    with col2:
        prediction_file = st.file_uploader("Upload Model Predictions (JSON)", type="json", key="pred")

    if golden_file and prediction_file:
        try:
            golden_data = json.load(golden_file)
            prediction_data = json.load(prediction_file)

            golden_fields = extract_all_fields_recursive(golden_data)
            prediction_fields = extract_all_fields_recursive(prediction_data)

            st.header("Select Fields to Compare")
            col1, col2 = st.columns(2)
            with col1:
                golden_field = st.selectbox("Golden Data Field", golden_fields)
            with col2:
                prediction_field = st.selectbox("Prediction Field", prediction_fields)

            if st.button("Calculate Scores", type="primary"):
                with st.spinner("Calculating scores..."):
                    def extract_values_by_field(obj, field_name):
                        results = []
                        def recurse(o):
                            if isinstance(o, dict):
                                if field_name in o:
                                    results.append(o[field_name])
                                for v in o.values():
                                    recurse(v)
                            elif isinstance(o, list):
                                for item in o:
                                    recurse(item)
                        recurse(obj)
                        return results

                    golden_texts = extract_values_by_field(golden_data, golden_field)
                    prediction_texts = extract_values_by_field(prediction_data, prediction_field)
                    num_items = min(len(golden_texts), len(prediction_texts))

                    results = []
                    all_text_pairs = []

                    for i in range(num_items):
                        g_text = golden_texts[i]
                        p_text = prediction_texts[i]
                        all_text_pairs.append({'index': i, 'golden_text': g_text, 'prediction_text': p_text})
                        if not g_text or not p_text:
                            continue
                        bleu_1, bleu_2, bleu_3, bleu_4 = calculate_bleu(g_text, p_text)
                        rouge1_f1, rouge2_f1, rougeL_f1 = calculate_rouge(g_text, p_text)
                        results.append({
                            'index': i,
                            'bleu_1': bleu_1,
                            'bleu_2': bleu_2,
                            'bleu_3': bleu_3,
                            'bleu_4': bleu_4,
                            'rouge1_f1': rouge1_f1,
                            'rouge2_f1': rouge2_f1,
                            'rougeL_f1': rougeL_f1
                        })

                    st.header("Text Pairs")
                    for i, pair in enumerate(all_text_pairs):
                        with st.expander(f"Index {pair['index']}", expanded=i == 0):
                            gc, pc = st.columns(2)
                            with gc:
                                st.subheader("Golden Text")
                                st.text_area("", pair['golden_text'], height=200, key=f"g_{i}")
                            with pc:
                                st.subheader("Prediction Text")
                                st.text_area("", pair['prediction_text'], height=200, key=f"p_{i}")

                    if results:
                        st.header("Comparison Scores")
                        df = pd.DataFrame([{
                            'Index': r['index'],
                            'BLEU-1': r['bleu_1'],
                            'BLEU-2': r['bleu_2'],
                            'BLEU-3': r['bleu_3'],
                            'BLEU-4': r['bleu_4'],
                            'ROUGE-1': r['rouge1_f1'],
                            'ROUGE-2': r['rouge2_f1'],
                            'ROUGE-L': r['rougeL_f1']
                        } for r in results])
                        st.dataframe(df)

                        st.subheader("Score Trends per Question (Line Chart)")
                        score_melted = df.melt(id_vars=["Index"],
                                               value_vars=["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L"],
                                               var_name="Metric", value_name="Score")
                        fig_line = px.line(score_melted, x="Index", y="Score", color="Metric", markers=True)
                        st.plotly_chart(fig_line, use_container_width=True)

                        st.header("Average Scores")
                        avg_df = pd.DataFrame({
                            'Metric': ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L'],
                            'Average Score': [
                                df['BLEU-1'].mean(),
                                df['BLEU-2'].mean(),
                                df['BLEU-3'].mean(),
                                df['BLEU-4'].mean(),
                                df['ROUGE-1'].mean(),
                                df['ROUGE-2'].mean(),
                                df['ROUGE-L'].mean()
                            ]
                        })
                        st.dataframe(avg_df)

                        st.subheader("Average Metric Scores (Bar Chart)")
                        fig_bar = px.bar(avg_df, x="Metric", y="Average Score", color="Metric", text="Average Score")
                        fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                        fig_bar.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                        st.plotly_chart(fig_bar, use_container_width=True)

                        st.download_button("Download Results as CSV", data=df.to_csv(index=False), file_name="comparison_scores.csv", mime="text/csv")

            if st.button("Calculate All Fields Scores", type="secondary"):
                with st.spinner("Calculating all common field scores..."):
                    common_fields = sorted(set(golden_fields).intersection(set(prediction_fields)))
                    all_field_results = []

                    def extract_values(obj, field_name):
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

                    for field in common_fields:
                        golden_texts = extract_values(golden_data, field)
                        prediction_texts = extract_values(prediction_data, field)
                        num_items = min(len(golden_texts), len(prediction_texts))

                        for i in range(num_items):
                            g_text = golden_texts[i]
                            p_text = prediction_texts[i]
                            if not g_text or not p_text:
                                continue
                            bleu_1, bleu_2, bleu_3, bleu_4 = calculate_bleu(g_text, p_text)
                            rouge1_f1, rouge2_f1, rougeL_f1 = calculate_rouge(g_text, p_text)
                            all_field_results.append({
                                'Field': field,
                                'Index': i,
                                'BLEU-1': bleu_1,
                                'BLEU-2': bleu_2,
                                'BLEU-3': bleu_3,
                                'BLEU-4': bleu_4,
                                'ROUGE-1': rouge1_f1,
                                'ROUGE-2': rouge2_f1,
                                'ROUGE-L': rougeL_f1
                            })

                    if all_field_results:
                        all_scores_df = pd.DataFrame(all_field_results)
                        st.subheader("Scores Across All Common Fields")
                        st.dataframe(all_scores_df)

                        st.subheader("Average Score per Field")
                        avg_by_field = all_scores_df.groupby("Field").mean(numeric_only=True).reset_index()
                        st.dataframe(avg_by_field)

                        st.download_button("Download All Fields Scores CSV", data=all_scores_df.to_csv(index=False),
                                           file_name="all_fields_scores.csv", mime="text/csv")
                        st.download_button("Download Average Score per Field", data=avg_by_field.to_csv(index=False),
                                           file_name="average_field_scores.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error processing files: {e}")
    else:
        st.info("Please upload both golden data and model prediction files to calculate scores.")

# ----------------------- TAB 2: Batch Evaluation -----------------------
with tab2:
    batch_metrics.run_batch_ui()

with tab3:
        st.header("Detailed Metrics Evaluation")

        # You can expand this section with batch processing functionality
        detailed_metrics.run_detailed_metrics()