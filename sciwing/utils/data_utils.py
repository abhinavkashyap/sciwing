from allennlp.data.dataset_readers.dataset_utils.span_utils import to_bioul
from typing import List
import wasabi


def convert_conll2003_ner_to_bioul(filename: str, out_filename: str):
    """ Converts the conll2003 file to bilou tagged strings
    and writes it to out_filename

    The out_filename will have the first column as word and
    the next three columns as the NER tags

    Parameters
    ----------
    filename: str
        Convert the file in conll2003 format to bioul tags
    out_filename: str
        Writes the file to bioul format

    Returns
    -------
    None

    """
    msg_printer = wasabi.Printer()
    lines: List[List[str]] = []
    labels: List[List[str]] = []

    with open(filename) as fp:
        lines_: List[str] = []
        labels_: List[str] = []  # every list is a label for one namespace
        for text in fp:
            text_ = text.strip()
            if bool(text_):
                line_labels = text_.split()
                line_ = line_labels[0]
                label_ = line_labels[3]  # all 3 tags
                lines_.append(line_)
                labels_.append(label_)
            elif text_ == "-DOCSTART-":
                # skip next empty line as well
                lines_ = []
                labels_ = []
                next(fp)
            else:
                if len(lines_) > 0 and len(labels_) > 0:
                    lines.append(lines_)
                    labels.append(labels_)
                    lines_ = []
                    labels_ = []
    bilou_tags = []
    for label in labels:
        bilou_ = to_bioul(tag_sequence=label, encoding="IOB1")
        bilou_tags.append(bilou_)

    with msg_printer.loading(f"writing BILOU tags for {filename}"):
        with open(out_filename, "w") as fp:
            for line, bilou_tags_ in zip(lines, bilou_tags):
                assert len(line) == len(bilou_tags_)
                for word, tag in zip(line, bilou_tags_):
                    fp.write(" ".join([word, tag, tag, tag]))
                    fp.write("\n")

                fp.write("\n")
    msg_printer.good(f"Finished writing BILOU tags for {filename}")


if __name__ == "__main__":
    import sciwing.constants as constants

    DATA_DIR = constants.PATHS["DATA_DIR"]
    import pathlib

    data_dir = pathlib.Path(DATA_DIR)
    train_conll = data_dir.joinpath("eng.train")
    convert_conll2003_ner_to_bioul(
        filename=str(train_conll), out_filename=data_dir.joinpath("conll_bioul.train")
    )

    dev_conll = data_dir.joinpath("eng.testa")
    convert_conll2003_ner_to_bioul(
        filename=str(dev_conll), out_filename=data_dir.joinpath("conll_bioul.dev")
    )

    test_conll = data_dir.joinpath("eng.testb")
    convert_conll2003_ner_to_bioul(
        filename=str(test_conll), out_filename=data_dir.joinpath("conll_bioul.test")
    )
