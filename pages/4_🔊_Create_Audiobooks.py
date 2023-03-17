import os
import pdfplumber
from gtts import gTTS
import streamlit as st

st.title("Create Audiobooks ~ convert a PDF document into an audiofile")

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