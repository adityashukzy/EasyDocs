import requests
import streamlit as st

st.title("Summarization ~ distil the essence of a text")

# ---------------------------------------------------------------------------- #

# Summarization function
def summarize(text, min_len, max_len, model_name="facebook/bart-large-cnn"):
    API_URL = "https://api-inference.huggingface.co/models/" + model_name
    headers = {"Authorization": "Bearer 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"}

    payload = \
        {
        "inputs": text,
        "parameters": {"min_length": min_len, "max_length": max_len, "do_sample": False}
        }
    response = requests.post(API_URL, headers=headers, json=payload)
    
    output_verbiage = 'generated_text' if model_name == 'adityashukzy/bart-base-finetuned-arxiv' else 'summary_text'
    summary = response.json()[0][output_verbiage]

    return summary

model_names = {
    "Facebook - BART-Large-CNN" : "facebook/bart-large-cnn",
    "EasyDocs - Finetuned BART" : "adityashukzy/bart-base-finetuned-arxiv"
    }


with st.expander("Keep in mind..."):
    st.markdown("1. For general-purpose texts, use bart-large-cnn.\n2. For academic or scientific texts, use bart-easydocs.\n3. The summary produced may not accurately cover all relevant parts of a text. Use this tool only as a starting guide.\n")

st.subheader("Enter text to summarize")
text = st.text_area(label="dont show", height=150, label_visibility="collapsed")

min_len_col, max_len_col, model_col = st.columns(3)

with min_len_col:
    min_len = st.slider("Select minimum number of words in summary", min_value=20, step=20, max_value=256, key='first', value=20)

with max_len_col:
    max_len = st.slider("Select maximum number of words in summary", min_value=20, step=20, max_value=256, key='second', value=100)

with model_col:
    verbose_model_name = st.selectbox("(Optional) Select model used for summarization", ("Facebook - BART-Large-CNN", "EasyDocs - Finetuned BART"))
    model_name = model_names[verbose_model_name]

with st.container():    
    if st.button("Click here to extract summary", use_container_width=True, type="primary"):
        with st.spinner("Summarizing..."):
            summary = summarize(text, min_len, max_len, model_name)
        
        st.markdown("---")

        if summary is not None:
            with st.expander("**Read Summary**", expanded=True):
                st.markdown(summary)