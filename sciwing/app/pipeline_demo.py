import streamlit as st
import requests
from sciwing.tokenizers.word_tokenizer import WordTokenizer
from spacy import displacy
import itertools

st.title("PDF Pipeline")

st.markdown(
    "Upload a pdf file and we will show you insights about its content. This feature is still "
    "in development. We will be providing more insights soon."
)

st.markdown("## Upload File")
content = st.file_uploader(
    label="Upload a research paper (Preferably a latex generated pdf)", type="pdf"
)

HTML_WRAPPER = """<div style="display:flex; align-content: center; justify-content: center; overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

if content is not None:
    with st.spinner("Extracting insights"):
        response = requests.post(
            f"http://localhost:8000/pdf_pipeline/uploadfile", files={"file": content}
        )
        json = response.json()
        abstract = json["abstract"]
        section_headers = json["section_headers"]
        normalized_section_headers = json["normalized_section_headers"]
        references = json["references"]
        parsed_reference_strings = json["parsed_reference_strings"]

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

        st.markdown("### Abstract")
        st.write(HTML_WRAPPER.format(abstract), unsafe_allow_html=True)

        header_normalized_header = [
            f"{header} ({normalized})"
            for header, normalized in zip(section_headers, normalized_section_headers)
        ]

        st.write("### Sections (Normalized Sections)")
        st.write(
            HTML_WRAPPER.format("<br />".join(header_normalized_header)),
            unsafe_allow_html=True,
        )

        st.write("### Parsed References. ")

        for reference, tags in zip(references, parsed_reference_strings):
            HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
            tokenizer = WordTokenizer(tokenizer="spacy-whitespace")
            doc = tokenizer.nlp(reference)

            # start index of every token
            token_indices = [token.idx for token in doc]

            # get start end index of every word
            start_end_indices = itertools.zip_longest(
                token_indices, token_indices[1:], fillvalue=len(reference)
            )
            start_end_indices = list(start_end_indices)

            ents = []
            for tag, (start_idx, end_idx) in zip(tags.split(), start_end_indices):
                ents.append({"start": start_idx, "end": end_idx, "label": tag})

            unique_tags = list(set(tags.split()))
            colors = {tag.upper(): next(colors_iter) for tag in unique_tags}
            options = {"ents": [tag.upper() for tag in unique_tags], "colors": colors}
            ex = [{"text": reference, "ents": ents, "title": "Entities"}]

            html = displacy.render(
                ex, style="ent", manual=True, options=options, page=False
            )
            html = html.replace("\n", " ")
            st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)


else:
    st.write(
        HTML_WRAPPER.format("Upload a PDF file to see results"), unsafe_allow_html=True
    )
