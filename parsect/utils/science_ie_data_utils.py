import pathlib
from typing import List, Dict, Any
import wasabi
import spacy
import parsect.constants as constants
from spacy.gold import biluo_tags_from_offsets

PATHS = constants.PATHS
DATA_DIR = PATHS["DATA_DIR"]


class ScienceIEDataUtils:
    """
        Science-IE is a SemEval Task that is aimed at extracting entities from scientific articles
        This class is a utility for various operations on the competitions data files
    """

    def __init__(self, folderpath: pathlib.Path, ignore_warnings=False):
        self.folderpath = folderpath
        self.ignore_warning = ignore_warnings
        self.entity_types = ["Process", "Material", "Task"]
        self.file_ids = self.get_file_ids()
        self.msg_printer = wasabi.Printer()
        self.nlp = spacy.load("en_core_web_sm")
        self._conll_col_sep = " "

    def get_file_ids(self) -> List[str]:
        file_ids = [file.stem for file in self.folderpath.iterdir()]
        file_ids = set(file_ids)
        file_ids = list(file_ids)
        return file_ids

    def get_text_from_fileid(self, file_id: str) -> str:
        path = self.folderpath.joinpath(f"{file_id}.txt")
        with open(path, "r") as fp:
            text = fp.readline()
            text = text.strip()

        return text

    def _get_annotations_for_entity(
        self, file_id: str, entity: str
    ) -> List[Dict[str, Any]]:
        """

        :param file_id: str
        :param entity: str
        One of Task Process or Material
        It filters through the annotation and returns only the annotation for entity
        :return:
        """
        annotations = []
        annotation_filepath = self.folderpath.joinpath(f"{file_id}.ann")
        with open(annotation_filepath, "r") as fp:
            for line in fp:
                if line.strip().startswith("T") and len(line.split("\t")) == 3:
                    entity_number, tag_start_end, words = line.split("\t")
                    if len(tag_start_end.split()) != 3:
                        self.msg_printer.warn(
                            f"Skipping LINE:{line} from file_id {file_id} for ENTITY:{entity}",
                            show=not self.ignore_warning,
                        )
                        continue
                    tag, start, end = tag_start_end.split()
                    start = int(start)
                    end = int(end)
                    if tag.lower() == entity.lower():
                        annotation = {
                            "start": start,
                            "end": end,
                            "words": words,
                            "entity_number": entity_number,
                            "tag": tag,
                        }
                        annotations.append(annotation)

        if len(annotations) == 0:
            self.msg_printer.warn(
                f"File {file_id} has 0 annotations for Type {entity}",
                show=not self.ignore_warning,
            )
        return annotations

    def get_bilou_lines_for_entity(self, file_id: str, entity: str):
        """
        Writes conll file for the entity type
        :param file_id: type str
        File id of the annotation file
        :param entity: type: str
        The entity for which conll file is written
        :return:
        """
        annotations = self._get_annotations_for_entity(file_id=file_id, entity=entity)
        text = self.get_text_from_fileid(file_id)

        return self._get_bilou_lines_for_entity(
            text=text, annotations=annotations, entity=entity
        )

    def _get_bilou_lines_for_entity(
        self, text: str, annotations: List[Dict[str, Any]], entity: str
    ) -> List[str]:
        entities = []
        for annotation in annotations:
            start = annotation["start"]
            end = annotation["end"]
            tag = annotation["tag"]
            entities.append((start, end, tag))

        doc = self.nlp(text)
        tags = biluo_tags_from_offsets(doc, entities)

        bilou_lines = []

        for token, tag in zip(doc, tags):
            if not token.is_space:
                bilou_line = f"{token.text}{self._conll_col_sep}{self._conll_col_sep.join([tag] * 3)}"
                bilou_lines.append(bilou_line)

        return bilou_lines

    def write_bilou_lines(
        self, out_filename: pathlib.Path, is_sentence_wise: bool = False
    ):
        filename_stem = out_filename.stem
        with self.msg_printer.loading(f"Writing BILOU Lines For ScienceIE"):
            for entity_type in self.entity_types:
                out_filename = pathlib.Path(
                    DATA_DIR, f"{filename_stem}_{entity_type.lower()}_conll.txt"
                )
                with open(out_filename, "w") as fp:
                    for file_id in self.file_ids:
                        # split the text into sentences and then write
                        if is_sentence_wise:
                            bilou_lines = self.get_sentence_wise_bilou_lines(
                                file_id=file_id, entity_type=entity_type
                            )
                        else:
                            bilou_lines = self.get_bilou_lines_for_entity(
                                file_id=file_id, entity=entity_type
                            )
                            bilou_lines = [bilou_lines]

                        for line in bilou_lines:
                            fp.write("\n".join(line))
                            fp.write("\n\n")

        self.msg_printer.good("Finished writing BILOU Lines For ScienceIE")

    def get_sentence_wise_bilou_lines(
        self, file_id: str, entity_type: str
    ) -> List[List[str]]:
        annotations = self._get_annotations_for_entity(
            file_id=file_id, entity=entity_type
        )
        text = self.get_text_from_fileid(file_id)

        entities = []
        for annotation in annotations:
            start = annotation["start"]
            end = annotation["end"]
            tag = annotation["tag"]
            entities.append((start, end, tag))

        doc = self.nlp(text)
        tags = biluo_tags_from_offsets(doc, entities)

        sentences = []

        current_sent = [f"{doc[0].text} {self._conll_col_sep.join([tags[0]] * 3)}"]
        for tag, token in zip(tags[1:], doc[1:]):
            if token.is_sent_start:
                sentences.append(current_sent)
                if not token.is_space:
                    current_sent = [
                        f"{token.text}{self._conll_col_sep}{self._conll_col_sep.join([tag] * 3)}"
                    ]
            else:
                if not token.is_space:
                    current_sent.append(
                        f"{token.text}{self._conll_col_sep}{self._conll_col_sep.join([tag] * 3)}"
                    )

        # finally add the last sentence
        sentences.append(current_sent)

        assert len(sentences) == len(list(doc.sents))

        return sentences

    def merge_files(
        self,
        task_filename: pathlib.Path,
        process_filename: pathlib.Path,
        material_filename: pathlib.Path,
        out_filename: pathlib.Path,
    ):
        with open(task_filename, "r") as task_fp, open(
            process_filename, "r"
        ) as process_fp, open(material_filename, "r") as material_fp, open(
            out_filename, "w"
        ) as out_fp:

            with self.msg_printer.loading("Merging Task Process and Material Files"):
                for task_line, process_line, material_line in zip(
                    task_fp, process_fp, material_fp
                ):
                    if bool(task_line.strip()):
                        word, _, _, task_tag = task_line.strip().split(
                            self._conll_col_sep
                        )
                        word, _, _, process_tag = process_line.strip().split(
                            self._conll_col_sep
                        )
                        word, _, _, material_tag = material_line.strip().split(
                            self._conll_col_sep
                        )
                        out_fp.write(
                            self._conll_col_sep.join(
                                [word, task_tag, process_tag, material_tag]
                            )
                        )
                        out_fp.write("\n")
                    else:
                        out_fp.write("\n")
            self.msg_printer.good("Finished Merging Task Process and Material Files")

    def get_sents(self, text: str):
        doc = self.nlp(text)
        sents = doc.sents
        sents = [sent.text for sent in sents]
        return sents


if __name__ == "__main__":
    import parsect.constants as constants

    PATHS = constants.PATHS
    FILES = constants.FILES
    SCIENCE_IE_TRAIN_FOLDER = FILES["SCIENCE_IE_TRAIN_FOLDER"]
    utils = ScienceIEDataUtils(
        folderpath=pathlib.Path(SCIENCE_IE_TRAIN_FOLDER), ignore_warnings=True
    )
    output_filename = pathlib.Path(DATA_DIR, "train.txt")
    # utils.write_bilou_lines(out_filename=output_filename, is_sentence_wise=False)

    utils.merge_files(
        pathlib.Path(DATA_DIR, "train_task_conll.txt"),
        pathlib.Path(DATA_DIR, "train_process_conll.txt"),
        pathlib.Path(DATA_DIR, "train_material_conll.txt"),
        pathlib.Path(DATA_DIR, "train_conll.txt"),
    )
