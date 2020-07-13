import streamlit as st
import requests
from spacy import displacy
import itertools
from sciwing.tokenizers.word_tokenizer import WordTokenizer


st.sidebar.subheader("NER Options")
model_selected = st.sidebar.radio(
    label="Select a Model",
    options=["Citation String Parsing", "I2B2 Clinical Notes Tagging"],
)


if model_selected == "Citation String Parsing":
    st.title("Citation String Parsing - Neural Parscit")
    st.markdown(
        "Neural Parscit is a citation parsing model. A citation string contains information "
        "like the author, the title of the publication, the conference/journal "
        "the year of publication etc. Neural Parscit extracts such information from references."
    )
    st.markdown(
        "**MODEL DESCRIPTION: ** SciWING trains a Bidirectional LSTM model "
        "with CRF Sequence tagging module that extracts information from a citation. The trained model "
        "on SciWING also includes an Elmo Embedder along with GloVe and character "
        "level embeddings."
    )
    st.markdown("---")

    st.write("**The Labels can be one of**")
    label_pill = """<span style="display:inline-block; border: 1px solid #0077B6; border-radius: 5px; padding: 5px; background-color: #0077B6; color: white; margin: 5px;">
                        {}
                </span>"""
    labels = [
        "author",
        "booktitle",
        "data",
        "editor",
        "institution",
        "journal",
        "location",
        "note",
        "pages",
        "publisher",
        "tech",
        "title",
        "volume",
    ]
    labels_legend = ""
    for label in labels:
        labels_legend += label_pill.format(label.upper())

    st.write(labels_legend, unsafe_allow_html=True)

    text_selected = st.selectbox(
        label="Select a Citation",
        options=[
            "Calzolari, N. (1982) Towards the organization of lexical definitions on a database structure. In E. Hajicova (Ed.), COLING '82 Abstracts, Charles University, Prague, pp.61-64.",
            "Caraballo, S.A. (1999) Automatic construction of a hypernym-labeled noun hierarchy. In Proceedings of the 37th Annual Meeting of the Association for Computational Linguistics (ACL'99), College Park, Maryland, pp. 120-126.",
        ],
    )

    user_text = st.text_input(label="Enter a citation string", value=text_selected)
    parse_button_clicked = st.button("Parse Citation")


elif model_selected == "I2B2 Clinical Notes Tagging":
    st.title("I2B2 Clinical Notes Tagging")
    st.markdown(
        "Clinical Natural Language Processing helps in identifying salient information from clinical notes."
        "Here, we have trained a neural network model on the **i2b2: Informatics for Integrating Biology and the Bedside** dataset."
        "This dataset has manual annotation for the problems identified, the treatments and tests suggested."
    )
    st.markdown(
        "**MODEL**: We trained a Bi-LSTM model with a CRF on top. We also included Elmo Embedding in the first layer"
    )
    text_selected = st.selectbox(
        label="Select an Example Clinical Note",
        options=[
            "Chest x - ray showed no evidence of cardiomegaly .",
            "Prostrate cancer and Renal failure",
            "Continue with Risperdal as per new psychiatrist",
        ],
    )
    st.markdown("---")
    user_text = st.text_input(label="Enter Clinical Notes", value=text_selected)
    parse_button_clicked = st.button("Parse Clinical Notes")

if parse_button_clicked:
    text_selected = user_text


with st.spinner(f"Please wait... Parsing Citation"):
    if model_selected == "Citation String Parsing":
        response = requests.get(f"http://localhost:8000/parscit/{text_selected}")
    elif model_selected == "I2B2 Clinical Notes Tagging":
        response = requests.get(f"http://localhost:8000/i2b2/{text_selected}")

json = response.json()
text = json["text_tokens"]
tags = json["tags"].split()

# tokenize the text using white space
tokenizer = WordTokenizer(tokenizer="spacy-whitespace")
doc = tokenizer.nlp(" ".join(text))

# start index of every token
token_indices = [token.idx for token in doc]

# get start end index of every word
start_end_indices = itertools.zip_longest(
    token_indices, token_indices[1:], fillvalue=len(" ".join(text))
)
start_end_indices = list(start_end_indices)


HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
ents = []
for tag, (start_idx, end_idx) in zip(tags, start_end_indices):
    ents.append({"start": start_idx, "end": end_idx, "label": tag})

# colors
unique_colors = [
    "#49483E",
    "#F92672",
    "#A6E22E",
    "#FD971F",
    "#66D9EF",
    "#AE81FF",
    "#A1EFE4",
    "#F8F8F2",
]
colors_iter = itertools.cycle(unique_colors)
unique_tags = list(set(tags))
colors = {tag.upper(): next(colors_iter) for tag in unique_tags}
options = {"ents": [tag.upper() for tag in unique_tags], "colors": colors}
ex = [{"text": " ".join(text), "ents": ents, "title": "Entities"}]

html = displacy.render(ex, style="ent", manual=True, options=options, page=False)
html = html.replace("\n", " ")
st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
