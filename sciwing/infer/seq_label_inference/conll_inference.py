from sciwing.infer.seq_label_inference.seq_label_inference import (
    SequenceLabellingInference,
)
import torch.nn as nn
from sciwing.data.datasets_manager import DatasetsManager
from typing import Union, Optional
import torch


class Conll2003Inference(SequenceLabellingInference):
    def __init__(
        self,
        model: nn.Module,
        model_filepath: str,
        datasets_manager: DatasetsManager,
        device: Optional[Union[str, torch.device]] = torch.device("cpu"),
        predicted_tags_namespace_prefix: str = "predicted_tags",
    ):
        super(Conll2003Inference, self).__init__(
            model=model,
            model_filepath=model_filepath,
            datasets_manager=datasets_manager,
            device=device,
            predicted_tags_namespace_prefix=predicted_tags_namespace_prefix,
        )

    def generate_predictions_for(
        self, task: str, test_filename: str, output_filename: str
    ):
        """
        Parameters
        ----------
        task: str
            Can be one of pos, dep or ner
            The task for which the predictions are made using the current model
        test_filename: str
            This is the eng.testb of the CoNLL 2003 dataset
        output_filename: str
            The file where you want to store predictions

        Returns
        -------
        None
            Writes the predictions to the output_filename

        The output file is meant to be used with conlleval.perl script
        ./conlleval < output_filename

        The file expects the correct tag and the predicted tag to be in the last
        two columns in that order
        The first column is the token for which the prediction is made
        """
        conll_labels = ["pos", "dep", "ner"]
        label_idx = conll_labels.index(task)

        lines: List[str] = []
        labels: List[str] = []
        with open(test_filename) as fp:
            lines_: List[str] = []  # contains the words
            labels_: List[str] = []  # contain true labels for the task
            for text in fp:
                text_ = text.strip()
                if bool(text_):
                    line_labels = text_.split()
                    line_ = line_labels[0]
                    label_ = line_labels[label_idx + 1]  # taking the right tag
                    lines_.append(line_)
                    labels_.append(label_)
                elif "-DOCSTART-" in text_:
                    # skip next empty line as well
                    next(fp)
                else:
                    if len(lines_) > 0 and len(labels_) > 0:
                        sentence = " ".join(lines_)
                        true_label = " ".join(labels_)
                        lines.append(sentence)
                        labels.append(true_label)
                        lines_ = []
                        labels_ = []

        with self.msg_printer.loading(
            f"Predicting {task.upper()} tags for {test_filename}"
        ):
            predictions = self.infer_batch(lines=lines)
            predictions = predictions[task.upper()]  # get predictions for the task

        self.msg_printer.good(f"Predicting {task.upper()} tags for {test_filename}")

        with self.msg_printer.loading(f"Writing Predictions to {output_filename}"):
            with open(output_filename, "w") as fp:
                for line, label, prediction in zip(lines, labels, predictions):
                    line_tokens = line.split()
                    label_tokens = label.split()
                    prediction_tokens = prediction.split()

                    for word, true_label, predicted_label in zip(
                        line_tokens, label_tokens, prediction_tokens
                    ):
                        fp.write(" ".join([word, true_label, predicted_label]))
                        fp.write("\n")

                    # \n before the next line starts
                    fp.write("\n")
        self.msg_printer.good(f"Finished Writing Predictions to {output_filename}")
