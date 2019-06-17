import json
import re
import wasabi
from pytablewriter import MarkdownTableWriter
import parsect.constants as constants
import os
import pandas as pd
import numpy as np
from deprecated import deprecated
import pathlib
from parsect.infer.bow_random_emb_lc_infer import get_random_emb_linear_classifier_infer
from parsect.infer.glove_emb_bow_linear_classifier_infer import (
    get_glove_emb_linear_classifier_infer,
)
from parsect.infer.elmo_emb_bow_linear_classifier_infer import (
    get_elmo_emb_linear_classifier_infer,
)
from parsect.infer.bert_emb_bow_linear_classifier_infer import (
    get_bert_emb_bow_linear_classifier_infer,
)
from parsect.infer.bi_lstm_lc_infer import get_bilstm_lc_infer
from parsect.infer.elmo_bi_lstm_lc_infer import get_elmo_bilstm_lc_infer

PATHS = constants.PATHS
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
REPORTS_DIR = PATHS["REPORTS_DIR"]


def generate_model_report(for_model: str, output_filename: str):
    output_dir_path = pathlib.Path(OUTPUT_DIR)
    all_fscores = {}
    row_names = None
    for dirname in output_dir_path.glob(f"{for_model}*"):
        infer = None
        if re.search("bow_random.*", for_model):
            infer = get_random_emb_linear_classifier_infer(dirname)
        if re.search("bow_glove.*", for_model):
            infer = get_glove_emb_linear_classifier_infer(dirname)
        if re.search("bow_elmo_emb.*", for_model):
            infer = get_elmo_emb_linear_classifier_infer(dirname)
        if re.search("bow_bert.*", for_model):
            infer = get_bert_emb_bow_linear_classifier_infer(dirname)
        if re.search("bow_scibert.*", for_model):
            infer = get_bert_emb_bow_linear_classifier_infer(dirname)
        if re.match("bi_lstm_lc.*", for_model):
            infer = get_bilstm_lc_infer(dirname)
        if re.match("elmo_bi_lstm_lc.*", for_model):
            infer = get_elmo_bilstm_lc_infer(dirname)

        fscores, row_names = infer.generate_report_for_paper()
        all_fscores[dirname.name] = fscores

    fscores_df = pd.DataFrame(all_fscores)
    fscores_df.index = row_names
    fscores_df.to_csv(output_filename, index=True, header=list(fscores_df.columns))


if __name__ == "__main__":
    # generate_model_report(
    #     for_model="bow_random_emb_lc",
    #     output_filename=os.path.join(REPORTS_DIR, "bow_random_report.csv"),
    # )
    # generate_model_report(
    #     for_model="bow_glove_emb_lc",
    #     output_filename=os.path.join(REPORTS_DIR, "bow_glove_report.csv")
    # )
    # generate_model_report(
    #     for_model="bow_elmo_emb_lc",
    #     output_filename=os.path.join(REPORTS_DIR, "bow_elmo_report.csv")
    # )
    # generate_model_report(
    #     for_model="bow_bert",
    #     output_filename=os.path.join(REPORTS_DIR, "bow_bert_report.csv")
    # )
    # generate_model_report(
    #     for_model="bow_scibert",
    #     output_filename=os.path.join(REPORTS_DIR, "bow_scibert_report.csv")
    # )
    # generate_model_report(
    #     for_model="bi_lstm_lc",
    #     output_filename=os.path.join(REPORTS_DIR, "bi_lstm_report.csv")
    # )
    generate_model_report(
        for_model="elmo_bi_lstm_lc",
        output_filename=os.path.join(REPORTS_DIR, "elmo_bi_lstm_lc.csv"),
    )
