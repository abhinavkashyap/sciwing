import streamlit as st
import requests

st.title("Citation Intent Classification")

st.markdown(
    "Identify the intent behind citing another scholarly document helps "
    "in fine-grain analysis of documents. Some citations refer to the "
    "methodology in another document, some citations may refer to other works"
    "for background knowledge and some might compare and contrast their methods with another work. "
    "Citation Intent Classification models classify such intents."
)
st.markdown(
    "**MODEL DESCRIPTION: ** This model is similar to [Arman Cohan et al](https://arxiv.org/pdf/1904.01608.pdf). We do not perform multi-task learning, but include "
    "ELMo Embeddings in the model."
)

st.markdown("---")

st.write("**The Labels can be one of: **")
st.write(
    """<span style="display:inline-block; border: 1px solid #0077B6; border-radius: 5px; padding: 5px; background-color: #0077B6; color: white; margin: 5px;">
        RESULT
    </span>
    <span style="display:inline-block; border: 1px solid #0077B6; border-radius: 5px; padding: 5px; background-color: #0077B6; color: white; margin: 5px;">
        BACKGROUND
    </span>
    <span style="display:inline-block; border: 1px solid #0077B6; border-radius: 5px; padding: 5px; background-color: #0077B6; color: white; margin: 5px;">
        METHOD
    </span>
    """,
    unsafe_allow_html=True,
)

text_selected = st.selectbox(
    label="Select a Citation",
    options=[
        "These results are in contrast with the findings of Santos et al.(16), who reported a significant association between low sedentary time and healthy CVF among Portuguese",
        "In order to improve ABM interventions and optimize symptom change, there has been a recent push to better understand the active ingredients and mechanisms that underlie post-ABM changes in symptoms (Beard 2011; Enock et al. 2014; Mogoaşe et al. 2014).",
    ],
)
user_text = st.text_input(label="Enter a citation", value=text_selected)
parse_button_clicked = st.button("Classify Citation")

if parse_button_clicked:
    text_selected = user_text

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

with st.spinner("Please wait... Classifying the Citation Intent"):
    response = requests.get(f"http://localhost:8000/cit_int_clf/{text_selected}")
json = response.json()
tag = json["tags"]
citation = json["citation"]
output_string = f"""<em>{citation}</em> &#8594;
<span style="display:inline-block; border: 1px solid #0077B6; border-radius: 5px; padding: 5px; background-color: #0077B6; color: white; margin: 5px;">
    Cites 
    <strong>{tag.upper()}</strong>
</span>"""
st.write(HTML_WRAPPER.format(output_string), unsafe_allow_html=True)
