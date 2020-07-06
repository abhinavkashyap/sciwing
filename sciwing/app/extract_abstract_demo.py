import streamlit as st
import requests


st.sidebar.title("SciWING-Extract Abstract")
st.sidebar.markdown("---")

st.title("Extract Abstracts from PDFs")
st.markdown("Extract the abstracts from your pdf.")

content = st.file_uploader(
    label="Upload a research paper (Preferably a latex generated pdf)", type="pdf"
)
HTML_WRAPPER = """<div style="display:flex; align-content: center; justify-content: center; overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

if content is not None:

    with st.spinner("Extracting the abstracts"):
        response = requests.post(
            f"http://localhost:8000/sectlabel/abstract", files={"file": content}
        )
        json = response.json()
        abstract = json["abstract"]

        st.markdown("### Abstract")

        st.write(HTML_WRAPPER.format(abstract), unsafe_allow_html=True)

else:
    st.write(HTML_WRAPPER.format("Upload a file"), unsafe_allow_html=True)
