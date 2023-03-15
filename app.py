import os
import fitz
import requests
import pdfplumber
import pytesseract
from gtts import gTTS
from PIL import Image
import streamlit as st
import pyperclip as pp
from zipfile import ZipFile

st.set_page_config(
    page_title="EasyDocs",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items=
    {
        'About': "# Welcome to EasyDocs📄 ! "
    }
)

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
    summary = response.json()[0]['summary_text']

    return summary

# OCR function
def extract_text(img, language='eng'):
    st.subheader("Extracted text")
    extracted_text = pytesseract.image_to_string(img, lang=language)
    return extracted_text

# Language Codes for OCR
lang_codes = {'English': 'eng', 'Hindi': 'hin', 'Tamil': 'tamil'}


# Splitting PDF doc into pages function
def retrieve_pages(pdf_file):
    text = ""
    i = 0
    list_of_img_paths = []
    mat = fitz.Matrix(2.0, 2.0)
    
    with fitz.open(stream = pdf_file.read(), filetype="pdf") as fl1:
        for pg in fl1:
            i = i+1
            pix = pg.get_pixmap(matrix=mat)
            # img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pth = os.path.join(os.getcwd(), "Page_" + str(i) + ".png")
            print(pth, end='\n\n')
            list_of_img_paths.append(pth)
            pix.save(pth)
    
    # Create a ZIP file containing all the images
    st.markdown(list_of_img_paths)
    with ZipFile("pdf_images.zip", "w") as zipObj:
        for img_path in list_of_img_paths:
            zipObj.write(img_path)
    
    return (os.path.exists('pdf_images.zip'))


# MAIN Function
def main():
    menu = ['Welcome', 'Summarize text', 'Extract text from an image (OCR)', 'Split PDF into pages', 'Create an audiobook for text/PDF']
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
                    with st.expander("**Read Summary**", expanded=True):
                        st.markdown(summary)


    elif option == 'Extract text from an image (OCR)':
        st.title("Transcribe text from an image 🔍")

        with st.expander("Keep in mind...", expanded=True):
            st.markdown("1. Upload any image by dragging and dropping or browsing your files.\n2. Copy or download the extracted txt.\n")

        uploader_col, lang_col = st.columns(2)

        with uploader_col:
            img_file = st.file_uploader("Upload your image containing text", type=['png','jpg'])

        with lang_col:
            language = st.selectbox("(Optional) Select language", ('English', 'Hindi', 'Tamil'))
        
        image_col, extracted_col = st.columns(2)


        if st.button("Click here to extract text", use_container_width=True, type="primary"):
            st.markdown("---")

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
                            with st.expander("**Read Extracted Text**", expanded=True):
                                st.markdown(content)
                        
                        copy_col, download_col = st.columns(2)

                        with copy_col:
                            if st.button("Copy to Clipboard"):
                                pp.copy(content)

                        with download_col:
                            st.download_button('Download extracted text', content)

    elif option == 'Split PDF into pages':
        st.title(" Split PDF document into its individual pages")

        with st.expander("Keep in mind...", expanded=True):
            st.markdown("1. We can split a PDF document of yours by extracting all its individual pages separately.\n2. You can then download a unified ZIP file containing all the pages in PNG format.\n")

        pdf_fl = st.file_uploader("Upload your PDF here", type=['pdf'])

        if st.button("Split PDF", use_container_width=True, type="primary"):
            st.markdown("---")
            if pdf_fl is not None:
                success = retrieve_pages(pdf_fl)

                if success is not None:
                    with open("pdf_images.zip", "rb") as fp:
                        st.download_button(
                            label="Download unified ZIP-file",
                            data=fp,
                            file_name="pdf_images.zip",
                            mime="application/zip"
                        )
                    
                    # delete file from local storage of VM running the app
                    if os.path.exists("pdf_images.zip"):
                        os.remove("pdf_images.zip")
    
    elif option == 'Create an audiobook for text/PDF':
        st.title('Create an audiobook for text/PDF')

        with st.expander("Keep in mind...", expanded=True):
            st.markdown("1. You can enter a text or upload a PDF to get the same in an audio MP3 format.\n")

        pdf_file = st.file_uploader("Upload your PDF here", type=['pdf'])
        slow = st.radio("Do you want it read out slowly?", ("Yes", "No"), index=1)

        if st.button("Create an Audiobook", use_container_width=True, type="primary"):
            st.markdown("---")
            
            if pdf_file is not None:
                pdf_text = "\n\n"

                try:
                    with pdfplumber.open(pdf_file) as pdf:
                        pages = pdf.pages
                        for i, val in enumerate(pages):
                            pdf_text += f"Page {i+1}\n\n" +  val.extract_text() + "\n\n\n"

                    print(pdf_text)
                    with st.spinner("Converting PDF to audio... "):
                        audio = gTTS(text=pdf_text, lang='en', slow=(True if slow == "Yes" else False), tld='co.in')
                        audio.save('audiobook.wav')
                    
                        st.audio('audiobook.wav', format='audio/wav')
                        os.remove('audiobook.wav')

                except:
                    st.error("PDF not in a readable format.")
    

if __name__ == "__main__":
    main()