import pytesseract
from PIL import Image
import streamlit as st

st.title("OCR ~ extract text from an image")


# OCR function
def extract_text(img, language='eng'):
    st.subheader("Extracted Text")
    extracted_text = pytesseract.image_to_string(img, lang=language)
    return extracted_text

# Language Codes for OCR
lang_codes = {'English': 'eng', 'Hindi': 'hin', 'Tamil': 'tam'}

with st.expander("Keep in mind...", expanded=True):
    st.markdown("1. Upload any image by dragging and dropping or browsing your files.\n2. Copy or download the extracted txt.\n")

uploader_col, lang_col = st.columns(2)

with uploader_col:
    img_file = st.file_uploader("Upload your image containing text", type=['png','jpg'])

with lang_col:
    language = st.selectbox("(Optional) Select language", ('English', 'Hindi', 'Tamil'))

btn = st.button("Click here to extract text", use_container_width=True, type="primary")
st.markdown("---")

if btn:
    image_col, extracted_col = st.columns(2)

    with image_col:
        if img_file is not None:
            # Show uploaded image
            img = Image.open(img_file)
            st.subheader('Uploaded Image')
            st.image(img)

    with extracted_col:
        if img_file is not None:
            with st.spinner("Extracting text..."):
                content = extract_text(img, lang_codes[language])
                if content is not None:
                    with st.expander("**Read Extracted Text**", expanded=True):
                        st.code(content, language=None)
                
                st.download_button('Download extracted text', content, use_container_width=True)
