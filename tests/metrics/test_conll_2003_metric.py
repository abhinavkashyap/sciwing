import pytest
from sciwing.datasets.seq_labeling.conll_dataset import CoNLLDatasetManager
import sciwing.constants as constants
from sciwing.metrics.conll_2003_metrics import ConLL2003Metrics
import pathlib

PATHS = constants.PATHS
DATA_DIR = PATHS["DATA_DIR"]


@pytest.fixture(scope="session")
def conll_dataset_manager():
    data_dir = pathlib.Path(DATA_DIR)
    train_filename = data_dir.joinpath("eng.train")
    dev_filename = data_dir.joinpath("eng.testa")
    test_filename = data_dir.joinpath("eng.testb")

    data_manager = CoNLLDatasetManager(
        train_filename=train_filename,
        dev_filename=dev_filename,
        test_filename=test_filename,
        train_only="ner",
        column_names=["POS", "DEP", "NER"],
    )
    return data_manager


@pytest.fixture(scope="session")
def setup_metric(conll_dataset_manager):
    data_manager = conll_dataset_manager
    metric = ConLL2003Metrics(datasets_manager=data_manager)
    lines, labels = data_manager.train_dataset.get_lines_labels()
    lines = lines[:2]
    labels = labels[:2]

    # max_label_length
    len_labels = [len(label.tokens["NER"]) for label in labels]
    max_label_len = max(len_labels)
    i_per_label = data_manager.namespace_to_vocab["NER"].get_idx_from_token("I-PER")

    # create dummy predictions
    predictions = []
    for i in range(len(lines)):
        predictions.append([i_per_label] * max_label_len)

    model_forward_dict = {"predicted_tags_NER": predictions}

    return metric, lines, labels, model_forward_dict


class TestConll2003Metric:
    def test_calc_metric(self, setup_metric):
        metric, lines, labels, model_forward_dict = setup_metric
        try:
            metric.calc_metric(
                lines=lines, labels=labels, model_forward_dict=model_forward_dict
            )
        except:
            pytest.fail("Calc metric for conll2003 failed")

    def test_get_metric(self, setup_metric):
        metric, lines, labels, model_forward_dict = setup_metric
        try:
            metric.calc_metric(
                lines=lines, labels=labels, model_forward_dict=model_forward_dict
            )
            metrics = metric.get_metric()
            assert "NER" in metrics.keys()
        except:
            pytest.fail("Get metric for conll2003 failed")

    def test_report_metric(self, setup_metric):
        metric, lines, labels, model_forward_dict = setup_metric
        try:
            metric.calc_metric(
                lines=lines, labels=labels, model_forward_dict=model_forward_dict
            )
            tables = metric.report_metrics()
            for namespace, table in tables.items():
                print(table)
        except:
            pytest.fail("Get metric for conll2003 failed")
