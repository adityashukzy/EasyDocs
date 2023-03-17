import os
import fitz
import streamlit as st
from zipfile import ZipFile

st.title("Split PDF ~ extract each page of a document")

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
            list_of_img_paths.append(pth)
            pix.save(pth)
    
    # Create a ZIP file containing all the images
    # st.markdown(list_of_img_paths)
    with ZipFile("pdf_images.zip", "w") as zipObj:
        for img_path in list_of_img_paths:
            zipObj.write(img_path)
    
    return (os.path.exists('pdf_images.zip'))


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