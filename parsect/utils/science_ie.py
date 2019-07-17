import pathlib
from typing import List, Dict, Any
import wasabi
from parsect.tokenizers.word_tokenizer import WordTokenizer
import parsect.constants as constants

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
        self.file_ids = self._get_file_ids()
        self.msg_printer = wasabi.Printer()
        self.word_tokenizer = WordTokenizer(tokenizer="spacy")

    def _get_file_ids(self) -> List[str]:
        file_ids = [file.stem for file in self.folderpath.iterdir()]
        file_ids = set(file_ids)
        file_ids = list(file_ids)
        return file_ids

    def _get_text(self, file_id: str) -> str:
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
        text = self._get_text(file_id)

        return self._get_bilou_lines_for_entity(
            text=text, annotations=annotations, entity=entity
        )

    def _get_bilou_lines_for_entity(
        self, text: str, annotations: List[Dict[str, Any]], entity: str
    ):
        # because we detect word boundaries only using space.
        # This is added to detect the last word in a proper manner
        text = text + " "
        start_tag_mapping = {}
        end_tag_mapping = {}
        for annotation in annotations:
            start = annotation["start"]
            end = annotation["end"]
            tag = annotation["tag"]
            start_tag_mapping[start] = tag
            end_tag_mapping[end] = tag

        annotation_word_starting: bool = False
        annotation_word_ending: bool = False
        text_word_start_index: int = 0
        current_tag = entity
        bilou_lines = []

        for idx, char in enumerate(text):
            # check whether there is an annotation start with this idx
            is_annotation_start = start_tag_mapping.get(idx, None)

            # found a annotation word starting
            if is_annotation_start:
                annotation_word_starting = True

            is_annotation_end = end_tag_mapping.get(idx, None)

            # this index is also the annotation end
            if is_annotation_end:
                annotation_word_ending = True

            # Found the end of the word in the text file
            # But no annotation word has started
            # The word that we found should represent an O-word
            if char == " " and annotation_word_starting == False:
                annotation_words = text[text_word_start_index:idx]
                lines = self._get_bilou_for_words(
                    annotation_words.split(), tag=current_tag, mark_as_O=True
                )
                bilou_lines.extend(lines)
                annotation_word_starting = False
                annotation_word_ending = False
                text_word_start_index = idx + 1

            # Found the end of the word
            # Also the annotation word has ended
            # The words found till now shoul represent words in the tag
            elif char == " " and annotation_word_ending == True:
                annotation_words = text[text_word_start_index:idx]
                lines = self._get_bilou_for_words(
                    annotation_words.split(), tag=current_tag, mark_as_O=False
                )
                bilou_lines.extend(lines)
                annotation_word_starting = False
                annotation_word_ending = False
                text_word_start_index = idx + 1

        num_bilou_lines = len(bilou_lines)
        num_text_words = len(text.split())
        assert (
            num_text_words == num_bilou_lines
        ), f"Number of Text Words {num_text_words}. Number of BILOU tagged words {num_bilou_lines}"
        return bilou_lines

    @staticmethod
    def _get_bilou_for_words(words: List[str], tag: str, mark_as_O: bool) -> List[str]:
        lines = []
        if mark_as_O:
            for word in words:
                line = f"{word} {' '.join(['O-'+tag] *3)}"
                lines.append(line)

        elif len(words) == 1:
            line = f"{words[0]} {' '.join(['U-' + tag] * 3)}"
            lines.append(line)

        else:
            for idx, word in enumerate(words):
                if idx == 0:
                    line = f"{word} {' '.join(['B-' + tag] * 3)}"
                elif idx == (len(words) - 1):
                    line = f"{word} {' '.join(['L-' + tag] * 3)}"
                else:
                    line = f"{word} {' '.join(['I-' + tag] * 3)}"
                lines.append(line)

        return lines

    def write_bilou_lines(self, out_filename: pathlib.Path):
        filename_stem = out_filename.stem
        with self.msg_printer.loading(f"Writing BILOU Lines For ScienceIE"):
            for entity_type in self.entity_types:
                out_filename = pathlib.Path(
                    DATA_DIR, f"{filename_stem}_{entity_type.lower()}_conll.txt"
                )
                with open(out_filename, "w") as fp:
                    for file_id in self.file_ids:
                        bilou_lines = self.get_bilou_lines_for_entity(
                            file_id=file_id, entity=entity_type
                        )
                        fp.write("\n".join(bilou_lines))
                        fp.write("\n \n")
        self.msg_printer.good("Finished writing BILOU Lines For ScienceIE")

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
                    if task_line.strip() != "":
                        word, _, _, task_tag = task_line.split()
                        word, _, _, process_tag = process_line.split()
                        word, _, _, material_tag = material_line.split()
                        out_fp.write(
                            " ".join([word, task_tag, process_tag, material_tag])
                        )
                        out_fp.write("\n")
                    else:
                        out_fp.write(task_line)
            self.msg_printer.good("Finished Merging Task Process and Material Files")
