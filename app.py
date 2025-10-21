import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="日本語対応 LLMデモ",  layout="centered")

st.title("日本語対応 文章分析")
st.write("日本語で入力できます")

# -------------------------
# 翻訳関数
# -------------------------
def translate_to_en(text):
    """日本語→英語翻訳"""
    try:
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en")
        return translator(text)[0]["translation_text"]
    except Exception as e:
        st.warning(f"翻訳モデルの読み込みに失敗しました。代わりに原文を使用します。({e})")
        return text

def translate_to_ja(text):
    """英語→日本語翻訳"""
    try:
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-jap")
        return translator(text)[0]["translation_text"]
    except Exception as e:
        st.warning(f"再翻訳モデルの読み込みに失敗しました。英語で出力します。({e})")
        return text


# -------------------------
# 🔹 Streamlit UI
# -------------------------
task = st.sidebar.selectbox(
    "機能を選択してください",
    ["感情分析", "要約", "質問応答", "翻訳テスト"]
)

# -------------------------
# 感情分析
# -------------------------
if task == "感情分析":
    st.subheader("感情分析（ポジティブ／ネガティブ）")
    jp_text = st.text_area("日本語のレビューを入力してください：")
    if st.button("分析する"):
        with st.spinner("分析中..."):
            en_text = translate_to_en(jp_text)
            model = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            result = model(en_text)[0]
            label_ja = "ポジティブ" if result["label"] == "POSITIVE" else "ネガティブ"
            st.success(f"結果：{label_ja}（スコア: {result['score']:.3f}）")

# -------------------------
# 要約
# -------------------------
elif task == "要約":
    st.subheader("要約（50〜55トークン）")
    jp_text = st.text_area("日本語の文章を入力してください：")
    if st.button("要約する"):
        with st.spinner("要約中..."):
            en_text = translate_to_en(jp_text)
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            summary_en = summarizer(en_text, max_length=55, min_length=50, do_sample=False)[0]["summary_text"]
            summary_ja = translate_to_ja(summary_en)
            st.success(summary_ja)

# -------------------------
# 質問応答
# -------------------------
elif task == "質問応答":
    st.subheader("質問応答")
    jp_context = st.text_area("文脈（テキスト）を入力してください：")
    jp_question = st.text_input("質問を入力してください：")
    if st.button("回答を取得"):
        with st.spinner("考えています..."):
            en_context = translate_to_en(jp_context)
            en_question = translate_to_en(jp_question)
            qa_model = pipeline("question-answering", model="deepset/minilm-uncased-squad2")
            answer_en = qa_model(question=en_question, context=en_context)["answer"]
            answer_ja = translate_to_ja(answer_en)
            st.success(answer_ja)

# -------------------------
# 翻訳テスト
# -------------------------
elif task == "翻訳テスト":
    st.subheader("🔹 翻訳テスト（日本語⇄英語）")
    jp_text = st.text_area("日本語を入力してください：")
    if st.button("翻訳する"):
        with st.spinner("翻訳中..."):
            en_text = translate_to_en(jp_text)
            back_ja = translate_to_ja(en_text)
            st.write("**英語訳:**", en_text)
            st.write("**再翻訳:**", back_ja)
