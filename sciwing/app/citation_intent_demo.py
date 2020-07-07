import streamlit as st
import requests

st.title("Citation Intent Classification")
st.markdown("---")
st.markdown(
    "Identify the intent behind citing another scholarly document helps"
    "in fine-grain analysis of documents. Some citations refer to the "
    "methodology used in another work, some citations may refer to other works"
    "for background work and some might compare their methods with another work."
    "Citation Intent Classification models classify such intents."
)

text_selected = st.selectbox(
    label="Select a Citation",
    options=[
        "These results are in contrast with the findings of Santos et al.(16), who reported a significant association between low sedentary time and healthy CVF among Portuguese",
        "The potential influence of accommodation (which could be driven through convergence accommodation) was eliminated by presenting the target through a pinhole (B1 mm) optically conjugate to the plane of the pupil [20].",
    ],
)
st.markdown("---")
user_text = st.text_input(label="Enter a citation", value=text_selected)
parse_button_clicked = st.button("Classify Citation")

if parse_button_clicked:
    text_selected = user_text

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

response = requests.get(f"http://localhost:8000/cit_int_clf/{text_selected}")
json = response.json()
tag = json["tags"]
citation = json["citation"]
output_string = f"""<em>{citation}</em> -> 
<span style="display:inline-block; border: 1px solid #0077B6; border-radius: 5px; padding: 5px; background-color: #0077B6; color: white; margin: 5px;">
    Cites 
    <strong>{tag.upper()}</strong>
</span>"""
st.write(HTML_WRAPPER.format(output_string), unsafe_allow_html=True)
