import json
import re
import wasabi
from pytablewriter import MarkdownTableWriter
import parsect.constants as constants
import os
import pandas as pd
import numpy as np

PATHS = constants.PATHS
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
REPORTS_DIR = PATHS["REPORTS_DIR"]


# TODO: This method is very specific to how logs
#   written in parsect engine module. There can
#   be better ways to log the results
def generate_report_from_test_log(log_filename: str,
                                  table_header: str) -> str:
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

        rows.append(['micro F1', results_dict["micro_fscore"]])
        rows.append(['macro F1', results_dict["macro_fscore"]])
        tbl_writer.value_matrix = rows
        tbl_string = tbl_writer.dumps()
        return tbl_string

    else:
        msg_printer.fail(f"Did not find a properly formed log " "for {log_filename}")


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
            log_filename = os.path.join(OUTPUT_DIR, directory, 'checkpoints', 'test.log')
            log_filenames.append(log_filename)
            all_models.append(directory)

    all_results = {}
    for model_name, log_filename in zip(all_models, log_filenames):
        all_results[model_name] = []
        with open(log_filename, 'r') as fp:
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

                all_results[model_name].append(results_dict['micro_fscore'])
                all_results[model_name].append(results_dict['macro_fscore'])

            else:
                msg_printer.fail(f"Did not find a properly formed log " "for {log_filename}")
                exit(1)

    results_df = pd.DataFrame(all_results)
    tbl_writer = MarkdownTableWriter()
    header = ['Class'] + list(results_df.columns)
    tbl_writer.headers = header
    values = results_df.values
    values = values.astype(np.str)
    indices = list(results_df.index)
    indices[-2] = 'Micro F1'
    indices[-1] = "Macro F1"
    values = np.insert(values, 0, indices, axis=1)
    tbl_writer.value_matrix = values.tolist()

    tbl_string = tbl_writer.dumps()
    return tbl_string


if __name__ == "__main__":

    bow_elmo_log_filename = os.path.join(
        OUTPUT_DIR, "bow_elmo_emb_lc_10e_1e-3lr", "checkpoints", "test.log"
    )
    bow_elmo_report_table_filename = os.path.join(
        REPORTS_DIR, "bow_elmo_emb_lc_10e_1e.md"
    )
    bow_elmo_tbl_string = generate_report_from_test_log(
        bow_elmo_log_filename,
        table_header="bow_elmo_emb_lc_10e_1e"
    )
    print(bow_elmo_tbl_string)

    random_emb_method_tbl_string = generate_report(for_model="bow_random_emb_lc")

    bow_random_emb_lc_filename = os.path.join(
        REPORTS_DIR, "bow_random_emb_lc.md"
    )

    with open(bow_random_emb_lc_filename, 'w') as fp:
        fp.write(random_emb_method_tbl_string)
