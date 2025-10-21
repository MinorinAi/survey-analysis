import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="日本語LLMデモ", layout="centered")

st.title("日本語テキスト分析デモ")

# Sidebar
task = st.sidebar.selectbox("機能を選択してください", ["要約", "質問応答"])

# 要約
elif task == "要約":
    st.subheader("要約")
    text = st.text_area("要約したい文章を入力してください：")

    # 長文要約関数
    def summarize_long_text(text, model_name="sonoisa/t5-base-japanese"):
        summarizer = pipeline("summarization", model=model_name)
        max_chunk_len = 400  # 一度に処理する文字数（400〜500が安全）

        # 長文を一定の長さごとに分割
        chunks = [text[i:i + max_chunk_len] for i in range(0, len(text), max_chunk_len)]

        # 各部分を要約してリストに格納
        summaries = []
        for i, chunk in enumerate(chunks):
            with st.spinner(f"部分 {i+1}/{len(chunks)} を要約中..."):
                summary = summarizer(
                    chunk, max_length=60, min_length=20, do_sample=False
                )[0]["summary_text"]
                summaries.append(summary)

        # すべての部分要約をまとめて最終要約
        joined = "。".join(summaries)
        final_summary = summarizer(
            joined, max_length=80, min_length=40, do_sample=False
        )[0]["summary_text"]
        return final_summary

    if st.button("要約する"):
        with st.spinner("長文を要約中です。しばらくお待ちください..."):
            summary = summarize_long_text(text)
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
