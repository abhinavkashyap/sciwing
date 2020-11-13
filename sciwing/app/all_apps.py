import streamlit as st
import os
import sys
import importlib.util
import pathlib


if __name__ == "__main__":
    # Parse command-line arguments.
    folder = pathlib.Path(".").absolute()

    # Get filenames for all files in this path, excluding this script.

    this_file = os.path.abspath(__file__)
    fnames = []

    for basename in folder.iterdir():
        fname = folder.joinpath(basename)
        if (
            fname.suffix == ".py"
            and "__init__.py" not in str(fname)
            and fname.stem != "all_apps"
            and fname.stem != "pipeline_demo"
        ):
            fnames.append(fname)
    fnames = sorted(fnames)
    # Make a UI to run different files.
    fnames_options_mapping = {
        "citation_intent_demo": "Citation Intent Classification",
        "scientific_articles_abstract": "Scientific Articles Abstract",
        "ner_demo": "Named Entity Recognition",
    }

    st.sidebar.image(
        "https://parsect-models.s3-ap-southeast-1.amazonaws.com/sciwing.png", width=250
    )
    st.sidebar.header("A Scientific Document Processing Toolkit.")

    st.sidebar.subheader("Applications")

    fname_to_run = st.sidebar.radio(
        "Select An application",
        fnames,
        format_func=lambda fname: fnames_options_mapping[fname.stem],
    )
    st.sidebar.markdown("---")

    # Create module from filepath and put in sys.modules, so Streamlit knows
    # to watch it for changes.

    fake_module_count = 0

    def load_module(filepath):
        global fake_module_count

        modulename = "_dont_care_%s" % fake_module_count
        spec = importlib.util.spec_from_file_location(modulename, filepath)
        module = importlib.util.module_from_spec(spec)
        sys.modules[modulename] = module

        fake_module_count += 1

    # Run the selected file.
    with open(fname_to_run) as f:
        load_module(fname_to_run)
        filebody = f.read()

    exec(filebody, {})

    st.sidebar.subheader("Contributions")
    st.sidebar.info(
        "We are open to contributions and suggestions. The project is available on "
        "[Github](https://github.com/abhinavkashyap/sciwing). Feel free to **contribute** by "
        "submitting a pull request or raising a feature request."
    )
    st.sidebar.subheader("About")
    st.sidebar.info(
        "This app is maintained by **WING-NUS** at the *National University of Singapore.* "
        "You can find more information about the app at [https://www.sciwing.io](https://www.sciwing.io)"
    )
