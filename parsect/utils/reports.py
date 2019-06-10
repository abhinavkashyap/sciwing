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
from parsect.infer.random_emb_bow_linear_classifier_infer import (
    get_random_emb_linear_classifier_infer,
)
from parsect.infer.glove_emb_bow_linear_classifier_infer import (
    get_glove_emb_linear_classifier_infer,
)
from parsect.infer.elmo_emb_bow_linear_classifier_infer import get_elmo_emb_linear_classifier_infer
from parsect.infer.bert_emb_bow_linear_classifier_infer import get_bert_emb_bow_linear_classifier_infer
from parsect.infer.bi_lstm_lc_infer import get_bilstm_lc_infer
from parsect.infer.elmo_bi_lstm_lc_infer import get_elmo_bilstm_lc_infer
PATHS = constants.PATHS
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
REPORTS_DIR = PATHS["REPORTS_DIR"]


# TODO: This method is very specific to how logs
#   written in parsect engine module. There can
#   be better ways to log the results


@deprecated(
    reason="This should never be used. Test log may not contain model with best params. This method will be removed"
)
def generate_report_from_test_log(log_filename: str, table_header: str) -> str:
    msg_printer = wasabi.Printer()
    tbl_writer = MarkdownTableWriter()
    tbl_writer.table_name = f"Report for {table_header}"
    tbl_writer.headers = ["Class #", "Fscore"]

    with open(log_filename, "r") as fp:
        output_results = json.load(fp)

    msg = output_results["msg"]
    match_obj = re.match("(.*) - (\{.*\})", msg)
    if match_obj:
        results = match_obj.groups()[1]
        results_dict = eval(results)
        fmeasure = results_dict["fscore"]
        classes = sorted(fmeasure.keys())

        rows = []
        for class_ in classes:
            row = [str(class_), fmeasure[class_]]
            rows.append(row)

        rows.append(["micro F1", results_dict["micro_fscore"]])
        rows.append(["macro F1", results_dict["macro_fscore"]])
        tbl_writer.value_matrix = rows
        tbl_string = tbl_writer.dumps()
        return tbl_string

    else:
        msg_printer.fail(f"Did not find a properly formed log " "for {log_filename}")


@deprecated(
    reason="Generates reporst from test logs. Test log may not contain model with best params. This method will be removed"
)
def generate_report(for_model="bow_random_emb_lc") -> str:
    """
    Generates the report for the whole model type
    :param for_model: type: str
    :return:
    """
    log_filenames = []
    msg_printer = wasabi.Printer()

    all_models = []

    for directory in os.listdir(OUTPUT_DIR):
        if directory.startswith(for_model):
            log_filename = os.path.join(
                OUTPUT_DIR, directory, "checkpoints", "test.log"
            )
            log_filenames.append(log_filename)
            all_models.append(directory.replace(for_model, ""))

    all_results = {}
    for model_name, log_filename in zip(all_models, log_filenames):
        all_results[model_name] = []
        with open(log_filename, "r") as fp:
            output_results = json.load(fp)
            msg = output_results["msg"]
            match_obj = re.match("(.*) - (\{.*\})", msg)
            if match_obj:
                results = match_obj.groups()[1]
                results_dict = eval(results)
                fmeasure = results_dict["fscore"]
                classes = sorted(fmeasure.keys())

                for class_ in classes:
                    all_results[model_name].append(fmeasure[class_])

                all_results[model_name].append(results_dict["micro_fscore"])
                all_results[model_name].append(results_dict["macro_fscore"])

            else:
                msg_printer.fail(
                    f"Did not find a properly formed log " "for {log_filename}"
                )
                exit(1)

    results_df = pd.DataFrame(all_results)
    tbl_writer = MarkdownTableWriter()
    header = ["Class"] + list(results_df.columns)
    tbl_writer.headers = header
    values = results_df.values
    values = values.astype(np.str)
    indices = list(results_df.index)
    indices[-2] = "Micro F1"
    indices[-1] = "Macro F1"
    values = np.insert(values, 0, indices, axis=1)
    tbl_writer.value_matrix = values.tolist()

    tbl_string = tbl_writer.dumps()
    return tbl_string


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
    fscores_df.to_csv(output_filename, index=True,
                      header=list(fscores_df.columns))


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
        output_filename=os.path.join(REPORTS_DIR, "elmo_bi_lstm_lc.csv")
    )

