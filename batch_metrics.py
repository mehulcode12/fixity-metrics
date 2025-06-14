import streamlit as st
import pandas as pd
import json
import re
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from io import StringIO
import plotly.express as px
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Ensure required data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

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

def run_batch_ui():
    st.header("Batch Upload: Multiple File Comparison")
    
    col1, col2 = st.columns(2)
    with col1:
        golden_files = st.file_uploader("Upload Multiple Golden JSON Files", type="json", accept_multiple_files=True, key="batch_golden")
    with col2:
        prediction_files = st.file_uploader("Upload Multiple Prediction JSON Files", type="json", accept_multiple_files=True, key="batch_pred")

    if golden_files and prediction_files:
        if len(golden_files) != len(prediction_files):
            st.warning("Please upload the same number of golden and prediction files.")
            return

        if st.button("Run Batch Evaluation"):
            with st.spinner("Processing files..."):
                batch_results = []

                for idx, (golden_file, pred_file) in enumerate(zip(golden_files, prediction_files)):
                    try:
                        golden_data = json.load(golden_file)
                        prediction_data = json.load(pred_file)

                        golden_fields = extract_all_fields_recursive(golden_data)
                        pred_fields = extract_all_fields_recursive(prediction_data)
                        common_fields = sorted(set(golden_fields).intersection(set(pred_fields)))

                        for field in common_fields:
                            golden_texts = extract_values(golden_data, field)
                            pred_texts = extract_values(prediction_data, field)
                            n = min(len(golden_texts), len(pred_texts))

                            for i in range(n):
                                g = golden_texts[i]
                                p = pred_texts[i]
                                if not g or not p:
                                    continue
                                bleu_1, bleu_2, bleu_3, bleu_4 = calculate_bleu(g, p)
                                rouge1, rouge2, rougeL = calculate_rouge(g, p)

                                batch_results.append({
                                    "Pair Index": idx + 1,
                                    "Golden File": golden_file.name,
                                    "Prediction File": pred_file.name,
                                    "Field": field,
                                    "Entry Index": i,
                                    "BLEU-1": bleu_1,
                                    "BLEU-2": bleu_2,
                                    "BLEU-3": bleu_3,
                                    "BLEU-4": bleu_4,
                                    "ROUGE-1": rouge1,
                                    "ROUGE-2": rouge2,
                                    "ROUGE-L": rougeL
                                })
                    except Exception as e:
                        st.error(f"Error processing pair {golden_file.name} & {pred_file.name}: {e}")

                if batch_results:
                    df = pd.DataFrame(batch_results)
                    st.success("Evaluation completed!")

                    st.subheader("Batch Evaluation Results")
                    st.dataframe(df)

                    # Existing: Overall average scores
                    st.subheader("Average Scores Across All Pairs")
                    avg_scores = df[["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L"]].mean().reset_index()
                    avg_scores.columns = ["Metric", "Average Score"]
                    st.dataframe(avg_scores)

                    # ✅ NEW: Average scores per file-pair per field
                    st.subheader("Average Scores Per File and Field")
                    field_wise_avg = df.groupby(["Golden File", "Prediction File", "Field"])[
                        ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L"]
                    ].mean().reset_index()
                    st.dataframe(field_wise_avg)

                    st.download_button("Download Batch Results CSV", data=df.to_csv(index=False),
                                       file_name="batch_scores.csv", mime="text/csv")
                    st.download_button("Download Average Scores CSV", data=avg_scores.to_csv(index=False),
                                       file_name="average_scores.csv", mime="text/csv")
                    st.download_button("Download Field-wise Averages CSV", data=field_wise_avg.to_csv(index=False),
                                       file_name="field_wise_scores.csv", mime="text/csv")
                    
                    # Set layout for 2 columns × 3 rows
                    col1, col2 = st.columns(2)

                    # ----------------------- Chart 1: Heatmap -----------------------
                    with col1:
                        st.subheader("1. BLEU-1 Heatmap (Fields × File Pair)")
                        heatmap_df = field_wise_avg.pivot_table(
                            index=["Golden File", "Prediction File"], 
                            columns="Field", 
                            values="BLEU-1"
                        )
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sns.heatmap(heatmap_df, annot=False, cmap="YlGnBu", cbar=True, ax=ax)
                        st.pyplot(fig)

                    # ----------------------- Chart 2: Grouped Bar (Field Avg Scores) -----------------------
                    with col2:
                        st.subheader("2. Field-wise Average Scores")
                        avg_by_field = field_wise_avg.groupby("Field")[["BLEU-1", "ROUGE-L"]].mean().reset_index()
                        avg_by_field_melted = avg_by_field.melt(id_vars="Field", var_name="Metric", value_name="Score")
                        chart = alt.Chart(avg_by_field_melted).mark_bar().encode(
                            x=alt.X('Field:N', sort=None),
                            y='Score:Q',
                            color='Metric:N',
                            column='Metric:N',
                            tooltip=["Field", "Metric", "Score"]
                        ).properties(width=150, height=300).interactive()
                        st.altair_chart(chart, use_container_width=True)

                    # ----------------------- Chart 3: Radar Plot -----------------------
                    with col1:
                        st.subheader("3. Radar Chart (File-wise Averages)")
                        import plotly.graph_objects as go
                        avg_pairwise = field_wise_avg.groupby(["Golden File", "Prediction File"])[
                            ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L"]
                        ].mean().reset_index()

                        categories = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L"]
                        fig = go.Figure()
                        for _, row in avg_pairwise.iterrows():
                            fig.add_trace(go.Scatterpolar(
                                r=row[categories].values,
                                theta=categories,
                                fill='toself',
                                name=f"{row['Golden File'].split('.')[0]}"
                            ))
                        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)

                    # ----------------------- Chart 4: BLEU-4 Line Trend -----------------------
                    with col2:
                        st.subheader("4. BLEU-1 Trend Across Fields")
                        fig = px.line(
                            field_wise_avg, 
                            x="Field", 
                            y="BLEU-1", 
                            color="Golden File",
                            markers=True
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)

                    # ----------------------- Chart 5: Box Plot (BLEU-4 by Field) -----------------------
                    
                    st.subheader("5. BLEU-1 Distribution by Field")
                    fig = px.box(field_wise_avg, x="Field", y="BLEU-1", color="Golden File",)
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)


    else:
        st.info("Please upload equal numbers of golden and prediction JSON files to proceed.")
