import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="日本語対応 文章分析", page_icon="🇯🇵", layout="centered")

st.title("日本語対応 文章分析")
st.write("日本語で入力できます")

# 翻訳モデル（日本語⇄英語）
to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en")
to_ja = pipeline("translation", model="Helsinki-NLP/opus-mt-en-jap")

task = st.sidebar.selectbox("機能を選択してください", ["感情分析", "要約", "質問応答", "英訳テスト"])

# --- 感情分析 ---
if task == "感情分析":
    st.subheader("感情分析（ポジティブ／ネガティブ）")
    jp_text = st.text_area("日本語のレビューを入力してください：")
    if st.button("分析する"):
        en_text = to_en(jp_text)[0]['translation_text']
        sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        result = sentiment_model(en_text)[0]
        label_ja = "ポジティブ" if result['label'] == "POSITIVE" else "ネガティブ"
        st.success(f"結果：{label_ja}（スコア: {result['score']:.3f}）")

# --- 要約 ---
elif task == "要約":
    st.subheader("要約")
    jp_text = st.text_area("日本語の文章を入力してください：")
    if st.button("要約する"):
        en_text = to_en(jp_text)[0]['translation_text']
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary_en = summarizer(en_text, max_length=55, min_length=50, do_sample=False)[0]["summary_text"]
        summary_ja = to_ja(summary_en)[0]['translation_text']
        st.write("要約結果：")
        st.success(summary_ja)

# --- 質問応答 ---
elif task == "質問応答":
    st.subheader("質問応答")
    jp_context = st.text_area("文脈（テキスト）を入力してください：")
    jp_question = st.text_input("質問を入力してください：")
    if st.button("回答を取得"):
        en_context = to_en(jp_context)[0]['translation_text']
        en_question = to_en(jp_question)[0]['translation_text']
        qa_model = pipeline("question-answering", model="deepset/minilm-uncased-squad2")
        answer_en = qa_model(question=en_question, context=en_context)["answer"]
        answer_ja = to_ja(answer_en)[0]['translation_text']
        st.success(answer_ja)

# --- 英訳テスト（確認用）---
elif task == "英訳テスト":
    st.subheader("翻訳確認")
    jp_text = st.text_area("日本語を入力してください：")
    if st.button("翻訳する"):
        en_text = to_en(jp_text)[0]['translation_text']
        back_ja = to_ja(en_text)[0]['translation_text']
        st.write("英語:", en_text)
        st.write("再翻訳:", back_ja)
