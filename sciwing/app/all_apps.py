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
        ):
            fnames.append(fname)

    # Make a UI to run different files.

    fname_to_run = st.sidebar.selectbox(
        "Select An application", fnames, format_func=lambda fname: fname.stem
    )

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
