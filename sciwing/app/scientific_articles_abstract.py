import streamlit as st
import requests


st.title("Extract Abstracts from PDFs")
st.markdown(
    "Marking the different logical sections of the paper is a fundamental step in analysing"
    "scientific documents. Here you can extract the abstracts from PDF documents."
)
st.markdown("**Model Description: This uses a Bi-LSTM with Elmo.**")
st.markdown("---")
st.markdown("## Upload File")
content = st.file_uploader(
    label="Upload a research paper (Preferably a latex generated pdf)", type="pdf"
)
HTML_WRAPPER = """<div style="display:flex; align-content: center; justify-content: center; overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

if content is not None:

    with st.spinner("Please wait... Extracting abstract."):
        response = requests.post(
            f"http://localhost:8000/sectlabel/abstract", files={"file": content}
        )
        json = response.json()
        abstract = json["abstract"]

        st.markdown("### Abstract")

        st.write(HTML_WRAPPER.format(abstract), unsafe_allow_html=True)

else:
    st.write(
        HTML_WRAPPER.format("No file uploaded. Please upload a file"),
        unsafe_allow_html=True,
    )
