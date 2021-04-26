from allennlp.data.dataset_readers.dataset_utils.span_utils import to_bioul
from typing import List, Dict
import wasabi
from collections import OrderedDict
from tqdm import tqdm
import json
import os


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


def intersect_conll_yago(conll_filename: str, yago_filename: str, out_filename: str):
    """ Get an intersection of the yago and conll documents . The two datasets are not aligned.
    We will consider only those documents that are aligned.

    Parameters
    ----------
    conll_filename : str
        The filename of the conll 2003 dataset
    yago_filename : str
        The filename where yago dataset is stored
    out_filename : str
        The filename which will contain the word NER tag and the corresponding yago entity
        If there is no entity then <None> will be used

    If train/dev/test files are passed for conll it should correspond to train/dev/test of
    the conll dataset

    Returns
    -------
    None
        Writes a file
    """

    printer = wasabi.Printer()
    # contains {"doc with words separated by space": "ner label separated by space"}
    conll_docs: Dict[str, str] = OrderedDict()
    with open(conll_filename) as fp:
        words = []
        labels = []
        for line in fp:
            line_ = line.strip()
            if bool(line_):
                line_labels = line_.split()
                word = line_labels[0]
                ner_label = line_labels[-1]
                words.append(word)
                labels.append(ner_label)

            elif "DOCSTART" in line:
                next(fp)
                continue

            else:
                if len(words) > 0:
                    assert len(words) == len(labels)
                    sentence = " ".join(words)
                    label = " ".join(labels)
                    words = []
                    labels = []
                    conll_docs[sentence] = label

    # contains {"doc with words separated by space": "yago entities separated by space"}
    yago_docs: Dict[str, str] = OrderedDict()
    with open(yago_filename) as fp:
        words = []
        yago_entities = []
        for line in fp:
            line_ = line.strip()
            if bool(line_):
                line_labels = line_.split()
                word = line_labels[0]
                if len(line_labels) == 7:
                    yago_entity = line_labels[3]  # the 4th column is the YAGO entity
                else:
                    yago_entity = "None"  # indicates there is no entity for this word

                words.append(word)
                yago_entities.append(yago_entity)

            elif "DOCSTART" in line:
                continue

            else:
                if len(words) > 0 and len(yago_entities) > 0:
                    assert len(words) == len(yago_entities)
                    sentence = " ".join(words)
                    entities = " ".join(yago_entities)
                    yago_docs[sentence] = entities
                    words = []
                    yago_entities = []

    # look for the keys that interesect with each other
    conll_keys = conll_docs.keys()
    yago_keys = yago_docs.keys()

    # set intersection
    intersecting_docs = conll_keys & yago_keys

    with open(out_filename, "w") as fp:
        for doc in tqdm(
            intersecting_docs,
            total=len(intersecting_docs),
            desc="Writing Intersection of CONLL and YAGO file",
        ):
            conll_ner_tags = conll_docs[doc]
            yago_entities = yago_docs[doc]

            # split them by space
            conll_ner_tags = conll_ner_tags.split()
            yago_entities = yago_entities.split()
            words = doc.split()

            for word, ner_tag, yago_tag in zip(words, conll_ner_tags, yago_entities):
                line = " ".join([word, ner_tag, yago_tag])
                fp.write(line)
                fp.write("\n")

            fp.write("\n")

    printer.good("Finished writing intersection of conll and yago files")


def write_scicite_to_sciwing_text_clf(scicite_json_filename: str, out_filename: str):
    """ SciCite files are jsonl filenames with citation strings.

    Parameters
    ----------
    scicite_json_filename : str
        The jsonl filename where citations are stored

    out_filename: str
        The output filename where the text classification dataset is stored

    Returns
    -------
    None

    """
    printer = wasabi.Printer()
    citations = []

    with printer.loading(f"Writing f{out_filename}"):
        with open(scicite_json_filename, "r") as fp:
            for line in fp:
                citation = json.loads(line)
                citations.append(citation)

        lines = []
        for citation in citations:
            citation_str = citation["string"].strip()
            citation_str = citation_str.replace("\n", " ")
            label_str = citation["label"].strip()
            label_str = label_str.replace("\n", " ")
            if bool(citation_str) and bool(label_str):
                line = "###".join([citation_str, label_str])
                lines.append(line)

        with open(out_filename, "w") as fp:
            for line in lines:
                fp.write(line)
                fp.write("\n")

    printer.good(f"Finished writing {out_filename}")


def write_pubmed_data_to_sciwing_seq2seq(
    pubmed_dir: str, subset: str, out_filename: str
):
    """ SciCite files are jsonl filenames with citation strings.

    Parameters
    ----------
    pubmed_dir : str
        The directory path to where pubmed dataset

    subset : str
        Choose from train, test and val

    out_filename : str
        Output file name

    Returns
    -------
    None

    """
    printer = wasabi.Printer()
    # inputs = []
    # abstracts = []
    lines = []

    text_dir = os.path.join(pubmed_dir, "inputs", subset)
    abstract_dir = os.path.join(pubmed_dir, "human-abstracts", subset)
    filename_list = [filename.split(".")[0] for filename in os.listdir(text_dir)]

    print(f"Reading pubmed {subset} data")
    for filename in tqdm(filename_list):
        with open(os.path.join(abstract_dir, f"{filename}.txt"), "r") as fp:
            abstract = fp.read()
            # abstracts.append(abstract)

        with open(os.path.join(text_dir, f"{filename}.json"), "r") as fp:
            input = json.load(fp)
            # inputs.append(input)

        abstract = abstract.strip().replace("\n", " ")
        text = " ".join([text["text"] for text in input["inputs"]])
        text = text.strip().replace("\n", " ")

        if bool(text) and bool(abstract):
            line = "###".join([text, abstract])
            lines.append(line)

    print(f"Writing pubmed {subset} data")
    with open(os.path.join(pubmed_dir, out_filename), "w") as fp:
        for line in lines:
            fp.write(line)
            fp.write("\n")

    printer.good(f"Finished writing {out_filename}")


if __name__ == "__main__":
    import sciwing.constants as constants
    import pathlib

    PATHS = constants.PATHS
    DATA_DIR = PATHS["DATA_DIR"]

    data_dir = pathlib.Path(DATA_DIR)
    # scicite_train_jsonl = data_dir.joinpath("scicite_train.jsonl")
    # scicite_dev_jsonl = data_dir.joinpath("scicite_dev.jsonl")
    # scicite_test_jsonl = data_dir.joinpath("scicite_test.jsonl")
    #
    # scicite_train_filename = data_dir.joinpath("scicite.train")
    # scicite_dev_filename = data_dir.joinpath("scicite.dev")
    # scicite_test_filename = data_dir.joinpath("scicite.test")
    #
    # write_scicite_to_sciwing_text_clf(
    #     scicite_json_filename=scicite_train_jsonl, out_filename=scicite_train_filename
    # )
    #
    # write_scicite_to_sciwing_text_clf(
    #     scicite_json_filename=scicite_dev_jsonl, out_filename=scicite_dev_filename
    # )
    #
    # write_scicite_to_sciwing_text_clf(
    #     scicite_json_filename=scicite_test_jsonl, out_filename=scicite_test_filename
    # )
    #
    pubmed_dir = data_dir.joinpath("pubmed")
    pubmed_train_filename = data_dir.joinpath("pubmedSeq2seq.train")
    pubmed_dev_filename = data_dir.joinpath("pubmedSeq2seq.dev")
    pubmed_test_filename = data_dir.joinpath("pubmedSeq2seq.test")

    write_pubmed_data_to_sciwing_seq2seq(pubmed_dir, "train", pubmed_train_filename)
    write_pubmed_data_to_sciwing_seq2seq(pubmed_dir, "val", pubmed_dev_filename)
    write_pubmed_data_to_sciwing_seq2seq(pubmed_dir, "test", pubmed_test_filename)
