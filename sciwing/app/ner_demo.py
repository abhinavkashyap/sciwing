import streamlit as st
import requests
from spacy import displacy
import itertools
from sciwing.tokenizers.word_tokenizer import WordTokenizer

st.sidebar.title("SciWING Demo")
st.sidebar.markdown("---")


st.title("Citation String Parsing - Neural Parscit")
st.markdown(
    "Neural Parscit is a citation parsing module. A citation string contains many information "
    "like the author, the title of the publication, the conference/journal the publication is "
    "submitted to, the year of publication. The Neural Parscit module is a Bidirectional LSTM model "
    "with CRF Sequence tagging module that extracts information from a citation. The trained model "
    "on SciWING also includes an Elmo Embedder along with a Glove embeddings and character "
    "level embeddings"
)

citation_selected = st.selectbox(
    label="Select a citation",
    options=[
        "Calzolari, N. (1982) Towards the organization of lexical definitions on a database structure. In E. Hajicova (Ed.), COLING '82 Abstracts, Charles University, Prague, pp.61-64.",
        "Caraballo, S.A. (1999) Automatic construction of a hypernym-labeled noun hierarchy. In Proceedings of the 37th Annual Meeting of the Association for Computational Linguistics (ACL'99), College Park, Maryland, pp. 120-126.",
    ],
)

st.markdown("---")
text_citation = st.text_input(label="Enter a citation string", value=citation_selected)
parse_citation_clicked = st.button("Parse Citation")

if parse_citation_clicked:
    citation_selected = text_citation


response = requests.get(f"http://localhost:8000/parscit/{citation_selected}")
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
