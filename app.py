import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="日本語LLMデモ", layout="centered")

st.title("日本語テキスト分析デモ")

# Sidebar
task = st.sidebar.selectbox("機能を選択してください", ["感情分析", "要約", "質問応答"])

# 感情分析
if task == "感情分析":
    st.subheader("感情分析")
    text = st.text_area("テキストを入力してください：")
    if st.button("分析する"):
        model = pipeline("sentiment-analysis", model="daigo/bert-base-japanese-sentiment")
        result = model(text)[0]
        label = result["label"]
        score = result["score"]
        st.success(f"感情：{label}（確信度: {score:.3f}）")

# 要約
elif task == "要約":
    st.subheader("要約")
    text = st.text_area("要約したい文章を入力してください：")
    if st.button("要約する"):
        summarizer = pipeline("summarization", model="sonoisa/t5-base-japanese")
        summary = summarizer(text, max_length=60, min_length=30, do_sample=False)[0]["summary_text"]
        st.write("要約結果:")
        st.success(summary)

# 質問応答
elif task == "質問応答":
    st.subheader("質問応答")
    context = st.text_area("文脈（テキスト）を入力：")
    question = st.text_input("質問を入力：")
    if st.button("回答を取得"):
        qa_model = pipeline("question-answering", model="izumi-lab/bert-small-japanese-finetuned-squad-v2")
        result = qa_model(question=question, context=context)
        st.success(f"回答：{result['answer']}")
