import questionary
from questionary.prompts.common import Choice
from parsect.commands.validators import is_valid_python_classname
from parsect.commands.validators import is_file_exist
import parsect.constants as constants
import jinja2
import pathlib
import autopep8
import wasabi


PATHS = constants.PATHS
TEMPLATES_DIR = PATHS["TEMPLATES_DIR"]
DATASETS_DIR = PATHS["DATASETS_DIR"]


def create_new_dataset_interactive():
    msg_printer = wasabi.Printer()

    dataset_name = questionary.text(
        "Name of Dataset? [Please provide a valid python ClassName]",
        qmark="?",
        validate=is_valid_python_classname,
    ).ask()

    dataset_type = questionary.select(
        "Chose the type of dataset you are creating?",
        choices=[
            Choice(title="Classification", value="classification"),
            Choice(title="Sequence Labeling", value="seq_labeling"),
        ],
        default="classification",
    ).ask()

    template_file: pathlib.Path = None

    if dataset_type == "classification":
        template_file = pathlib.Path(
            TEMPLATES_DIR, "classification_dataset_template.txt"
        )

    with open(template_file, "r") as fp:
        template_str = "".join(fp.readlines())

    template_variables = {
        "className": dataset_name,
        "word_embedding_type": "random",
        "word_embedding_dimension": 100,
        "word_start_token": "<SOS>",
        "word_end_token": "<EOS>",
        "word_pad_token": "<PAD>",
        "word_unk_token": "<UNK>",
        "train_size": 0.8,
        "test_size": 0.2,
        "validation_size": 0.5,
    }

    # Code generation
    template = jinja2.Template(template_str)
    class_code = template.render(template_variables)
    class_code = autopep8.fix_code(class_code)
    out_filepath = pathlib.Path(DATASETS_DIR, "classification", f"{dataset_name}.py")
    with open(out_filepath, "w") as fp:
        fp.write(class_code)
    msg_printer.good(f"Created a new dataset file in {str(out_filepath)}")
