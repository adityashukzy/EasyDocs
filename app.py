import requests
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

# MAIN Function
def main():
    menu = ['Welcome', 'Summarize text']
    with st.sidebar.expander("Menu", expanded=False):
        option = st.selectbox('Choose your task', menu)
        st.subheader("Made with ❤️ by Team Agnes")

    if option == 'Welcome':
        # img_path = "dependencies/black-bg-logo.gif"

        # st.image(img_path,width = 500)
        st.subheader("EasyDocs is a one-stop solution combining all your most needed tools as a student. We understand the value and preciousness of time and that is why we have aimed to make EasyDocs as no-nonsense as possible!")

        st.write("👉 Summarize webpages and long text documents and breeze through the essentials!")
        st.write("👉 Scan images for text and have them transcribed. This is called Optical Character Recognition!")
        st.write("👉 Play around with and explore the extremely convenient method of learning: *audiobooks*!")
        st.write("👉 If you're just looking around, maybe visit EzPz our chatbot and learn about EasyDocs!")

    elif option == 'Summarize text':
        ## URL summarization
        st.title("Summarize any academic text!")
        st.write("Enter a text below and choose how many sentences long you want the summary to be, give it a moment and enjoy! We recognize the value that a short and crisp summary can have when it comes to skimming through pages and pages of dense information and are here to make that experience a little less painful for you!")
        with st.expander("Keep in mind..."):
            st.write("For general purposes, use bart-large-cnn.")


        text = st.text_area("Text to summarize")

        min_len_col, max_len_col, model_col = st.columns(3)

        with min_len_col:
            min_len = st.slider("Select minimum number of words in summary", min_value=20, step=20, max_value=256, key='first', value=20)

        with max_len_col:
            max_len = st.slider("Select maximum number of words in summary", min_value=20, step=20, max_value=256, key='second', value=100)

        with model_col:
            model_name = st.selectbox("(Optional) Select model used for summarization", ("facebook/bart-large-cnn", "adityashukzy/bart-easydocs"))

        with st.container():    
            if st.button("Extract summary", use_container_width=True):
                with st.spinner("Summarizing..."):
                    summary = summarize(text, min_len, max_len, model_name)
                
                st.subheader(summary)


if __name__ == "__main__":
    main()