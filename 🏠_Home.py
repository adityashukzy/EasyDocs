import streamlit as st

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

# MAIN Function
def main():
    st.subheader("EasyDocs is a one-stop solution combining all your most needed tools as a student. We understand the value and preciousness of time and that is why we have aimed to make EasyDocs as no-nonsense as possible!")

    st.write("👉 Summarize webpages and long text documents and breeze through the essentials!")
    st.write("👉 Scan images for text and have them transcribed. This is called Optical Character Recognition!")
    st.write("👉 Play around with and explore the extremely convenient method of learning: *audiobooks*!")
    st.write("👉 If you're just looking around, maybe visit EzPz our chatbot and learn about EasyDocs!")


if __name__ == "__main__":
    main()