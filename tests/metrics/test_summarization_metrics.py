import pytest
from sciwing.metrics.summarization_metrics import SummarizationMetrics
from sciwing.datasets.summarization.abstractive_text_summarization_dataset import (
    AbstractiveSummarizationDatasetManager,
)
from sciwing.data.line import Line


@pytest.fixture(scope="session")
def abs_sum_dataset_manager(tmpdir_factory):
    train_file = tmpdir_factory.mktemp("train_data").join("train.txt")
    train_file.write(
        "word11_train word21_train###word11_label word21_label\nword12_train word22_train word32_train###word11_label word22_label"
    )

    dev_file = tmpdir_factory.mktemp("dev_data").join("dev.txt")
    dev_file.write(
        "word11_dev word21_dev###word11_label word21_label\nword12_dev word22_dev word32_dev###word11_label word22_label"
    )

    test_file = tmpdir_factory.mktemp("test_data").join("test.txt")
    test_file.write(
        "word11_test word21_test###word11_label word21_label\nword12_test word22_test word32_test###word11_label word22_label"
    )

    data_manager = AbstractiveSummarizationDatasetManager(
        train_filename=str(train_file),
        dev_filename=str(dev_file),
        test_filename=str(test_file),
    )

    return data_manager


@pytest.fixture
def setup_scorer(abs_sum_dataset_manager):
    dataset_manager = abs_sum_dataset_manager
    scorer = SummarizationMetrics(dataset_manager)

    lines = [
        Line("word11_train word21_train"),
        Line("word12_train word22_train word32_train"),
    ]
    true_summary = [
        Line("word11_label word21_label"),
        Line("word11_label word22_label"),
    ]
    true_summary_tokens = ["word11_label", "word22_label", "word33_label"]
    pred_summary_tokens = [
        "word11_label",
        "word22_label",
        "word23_label",
        "word33_label",
    ]
    predicted_tags = {"predicted_tags_tokens": [[0, 2], [1, 4, 5]]}
    return (
        scorer,
        (lines, true_summary, predicted_tags),
        (true_summary_tokens, pred_summary_tokens),
    )


class TestSummarizationMetrics:
    def test_rouge_n(self, setup_scorer):
        scorer, _, (true_summary_tokens, pred_summary_tokens) = setup_scorer
        rouge_1 = scorer._rouge_n(true_summary_tokens, pred_summary_tokens, 1)
        rouge_2 = scorer._rouge_n(true_summary_tokens, pred_summary_tokens, 2)
        rouge_l = scorer._rouge_l(true_summary_tokens, pred_summary_tokens)
        print(rouge_l)
        assert rouge_2 == 0.4
        assert rouge_1 > 0.8
        assert rouge_l > 0.8

    def test_scorer(self, setup_scorer):
        scorer, (lines, true_summary, predicted_tags), _ = setup_scorer
        scorer.calc_metric(lines, true_summary, predicted_tags)
        metrics = scorer.get_metric()
        print(metrics)
        assert False
