import requests
import pytesseract
from PIL import Image
import streamlit as st

st.set_page_config(
    page_title="EasyDocs",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        # 'Get Help': 'https://www.extremelycoolapp.com/help',
        # 'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# Welcome to EasyDocs📄 ! "
    }
)


# Loading (& caching) the model
@st.cache_resource
def load_model(model_name):
    from transformers import pipeline
    model = pipeline("summarization", model=model_name)
    return model

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
    summary = response.json()[0]['summary_text']

    return summary

# OCR function
def extract_text(img, language='eng'):
    st.subheader("Extracted text")
    extracted_text = pytesseract.image_to_string(img, lang=language)
    return extracted_text


# Language Codes for OCR
lang_codes = {'English': 'eng', 'Hindi': 'hin', 'Tamil': 'tamil'}

# MAIN Function
def main():
    menu = ['Welcome', 'Summarize text', 'Extract text from an image (OCR)']
    with st.sidebar.expander("Menu", expanded=False):
        option = st.selectbox('Choose your task', menu)

    if option == 'Welcome':
        st.subheader("EasyDocs is a one-stop solution combining all your most needed tools as a student. We understand the value and preciousness of time and that is why we have aimed to make EasyDocs as no-nonsense as possible!")

        st.write("👉 Summarize webpages and long text documents and breeze through the essentials!")
        st.write("👉 Scan images for text and have them transcribed. This is called Optical Character Recognition!")
        st.write("👉 Play around with and explore the extremely convenient method of learning: *audiobooks*!")
        st.write("👉 If you're just looking around, maybe visit EzPz our chatbot and learn about EasyDocs!")

    elif option == 'Summarize text':
        ## URL summarization
        st.title("Summarize any text!")
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
            model_name = st.selectbox("(Optional) Select model used for summarization", ("facebook/bart-large-cnn", "adityashukzy/bart-easydocs"))

        with st.container():    
            if st.button("Click here to extract summary", use_container_width=True, type="primary"):
                with st.spinner("Summarizing..."):
                    summary = summarize(text, min_len, max_len, model_name)
                
                st.markdown("---")

                if summary is not None:
                    with st.expander("**Read Summary**", ):
                        st.markdown(summary)


    elif option == 'Extract text from an image (OCR)':
        st.title("Transcribe text from an image 🔍")

        with st.expander("Keep in mind..."):
            st.markdown("1. Upload any image by dragging and dropping or browsing your files.\n2. Copy or download the extracted txt.\n")

        uploader_col, lang_col = st.columns(2)

        with uploader_col:
            img_file = st.file_uploader("Upload your image containing text", type=['png','jpg'])

        with lang_col:
            language = st.selectbox("(Optional) Select language", ('English', 'Hindi', 'Tamil'))
        
        image_col, extracted_col = st.columns(2)

        with image_col:
            if img_file is not None:
                # Show uploaded image
                img = Image.open(img_file)
                st.subheader('Uploaded Image:')
                st.image(img)

        with extracted_col:
            if img_file is not None:
                with st.spinner("Extracting text..."):
                    content = extract_text(img, lang_codes[language])
                    if content is not None:
                        with st.expander("**Read Extracted Text**", ):
                            st.markdown(content)
                    
                    st.download_button('Download extracted text', content)

if __name__ == "__main__":
    main()