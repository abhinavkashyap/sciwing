import questionary
from questionary.prompts.common import Choice
from parsect.commands.validators import is_valid_python_classname
from parsect.commands.validators import is_file_exist
import parsect.constants as constants
import jinja2
import pathlib
import autopep8
import wasabi
from typing import Dict, Optional, Any


PATHS = constants.PATHS
TEMPLATES_DIR = PATHS["TEMPLATES_DIR"]
DATASETS_DIR = PATHS["DATASETS_DIR"]


class ClassificationDatasetGenerator(object):
    def __init__(self, dataset_name: str, filename: Optional[str] = None):
        self.dataset_name = dataset_name
        self.template_file = pathlib.Path(
            TEMPLATES_DIR, "classification_dataset_template.txt"
        )
        self.template = self._get_template()
        self.filename = filename
        self.template_variables = self.interact()

    def _get_template(self):
        with open(self.template_file, "r") as fp:
            template_string = "".join(fp.readlines())
        jinja_template = jinja2.Template(template_string)
        return jinja_template

    def generate(self):
        with open(
            pathlib.Path(DATASETS_DIR, "classification", f"{self.dataset_name}.py"), "w"
        ) as fp:
            rendered_class = self.template.render(self.template_variables)
            rendered_class = autopep8.fix_code(rendered_class)
            fp.write(rendered_class)

    def interact(self) -> Dict[str, Any]:

        debug = questionary.confirm(
            "Do you want default value for debug to be True?: ", default=False
        ).ask()

        debug_dataset_proportion = questionary.text(
            message="Enter Proportion of dataset for debug: ", default="0.1"
        ).ask()
        debug_dataset_proportion = float(debug_dataset_proportion)

        word_embedding_type = questionary.select(
            message="Chose one of the embeddings available: ",
            choices=[
                Choice(title="random", value="random"),
                Choice(title="parscit", value="parscit"),
                Choice(title="glove_6B_100", value="glove_6B_100"),
                Choice(title="glove_6B_200", value="glove_6B_200"),
                Choice(title="glove_6B_300", value="glove_6B_300"),
            ],
        ).ask()

        embeddingtype2embedding_dimension = {
            "glove_6B_100": 100,
            "glove_6B_200": 200,
            "glove_6B_300": 300,
            "parscit": 500,
        }

        if word_embedding_type == "random":
            random_emb_dim = questionary.text(
                message="Enter embedding dimension: ", default="100"
            ).ask()
            random_emb_dim = int(random_emb_dim)
            embeddingtype2embedding_dimension["random"] = random_emb_dim

        emb_dim = embeddingtype2embedding_dimension[word_embedding_type]

        word_start_token = questionary.text(
            message="Enter default token to be used for beginning of sentence: ",
            default="<SOS>",
        ).ask()

        word_end_token = questionary.text(
            message="Enter default token to be used for end of sentence: ",
            default="<EOS>",
        ).ask()

        word_pad_token = questionary.text(
            message="Enter default token to be used for padding sentences: ",
            default="<PAD>",
        ).ask()

        word_unk_token = questionary.text(
            message="Enter default token to be used in case the word is not found in the vocab: ",
            default="<UNK>",
        ).ask()

        train_size = questionary.text(
            message="Enter default size of the dataset that will be used for training: ",
            default="0.8",
        ).ask()
        train_size = float(train_size)

        test_size = questionary.text(
            message="Enter default size of the dataset that will be used for testing: ",
            default="0.2",
        ).ask()

        validation_size = questionary.text(
            message="Enter default size fo the dataset that will be used for validation.: "
            "This will be the proportion of the test size that will be used",
            default="0.5",
        ).ask()

        tokenizer_type = questionary.select(
            message="What is the default tokenization that you would want to use?: ",
            choices=[
                Choice(
                    title="Vanilla (sentences are separated by space to form words)",
                    value="vanilla",
                ),
                Choice(title="Spacy tokenizer", value="spacy"),
            ],
            default="vanilla",
        ).ask()

        word_tokenizer = f"WordTokenizer(tokenizer={tokenizer_type})"

        template_options_dict = {
            "className": self.dataset_name,
            "word_embedding_type": word_embedding_type,
            "word_embedding_dimension": emb_dim,
            "word_start_token": word_start_token,
            "word_end_token": word_end_token,
            "word_pad_token": word_pad_token,
            "word_unk_token": word_unk_token,
            "train_size": train_size,
            "validation_size": validation_size,
            "test_size": test_size,
            "word_tokenization_type": tokenizer_type,
            "word_tokenizer": word_tokenizer,
        }
        return template_options_dict


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

    if dataset_type == "classification":
        dataset_generator = ClassificationDatasetGenerator(dataset_name=dataset_name)
        dataset_generator.generate()
