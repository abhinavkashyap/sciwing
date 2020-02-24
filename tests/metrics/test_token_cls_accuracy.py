import pytest
from sciwing.metrics.token_cls_accuracy import TokenClassificationAccuracy
from sciwing.datasets.seq_labeling.seq_labelling_dataset import (
    SeqLabellingDatasetManager,
)
from sciwing.utils.class_nursery import ClassNursery


@pytest.fixture(scope="session")
def seq_dataset_manager(tmpdir_factory):
    train_file = tmpdir_factory.mktemp("train_data").join("train.txt")
    train_file.write("word11_train###label1 word21_train###label2")

    dev_file = tmpdir_factory.mktemp("dev_data").join("dev.txt")
    dev_file.write(
        "word11_dev###label1 word21_dev###label2\nword12_dev###label1 word22_dev###label2 word32_dev###label3"
    )

    test_file = tmpdir_factory.mktemp("test_data").join("test.txt")
    test_file.write(
        "word11_test###label1 word21_test###label2\nword12_test###label1 word22_test###label2 word32_test###label3"
    )

    data_manager = SeqLabellingDatasetManager(
        train_filename=str(train_file),
        dev_filename=str(dev_file),
        test_filename=str(test_file),
    )

    return data_manager


@pytest.fixture
def setup_basecase(seq_dataset_manager):
    predicted_tags = [[5, 4]]
    data_manager = seq_dataset_manager
    lines, labels = data_manager.train_dataset.get_lines_labels()

    expected_precision = {5: 1.0, 4: 1.0}
    expected_recall = {5: 1.0, 4: 1.0}
    expected_fmeasure = {5: 1.0, 4: 1.0}
    expected_macro_precision = 1.0
    expected_macro_recall = 1.0
    expected_macro_fscore = 1.0
    expected_num_tps = {5: 1.0, 4: 1.0}
    expected_num_fps = {5: 0.0, 4: 0.0}
    expected_num_fns = {5: 0.0, 4: 0.0}
    expected_micro_precision = 1.0
    expected_micro_recall = 1.0
    expected_micro_fscore = 1.0

    token_cls_metric = TokenClassificationAccuracy(datasets_manager=data_manager)

    model_forward_dict = {"predicted_tags_seq_label": predicted_tags}

    return (
        token_cls_metric,
        lines,
        labels,
        model_forward_dict,
        {
            "expected_precision": expected_precision,
            "expected_recall": expected_recall,
            "expected_fscore": expected_fmeasure,
            "expected_macro_precision": expected_macro_precision,
            "expected_macro_recall": expected_macro_recall,
            "expected_macro_fscore": expected_macro_fscore,
            "expected_num_tps": expected_num_tps,
            "expected_num_fps": expected_num_fps,
            "expected_num_fns": expected_num_fns,
            "expected_micro_precision": expected_micro_precision,
            "expected_micro_recall": expected_micro_recall,
            "expected_micro_fscore": expected_micro_fscore,
        },
    )


class TestTokenClsAccuracy:
    def test_base_case_get_metric(self, setup_basecase):
        metric, lines, labels, model_forward_dict, expected = setup_basecase
        metric.calc_metric(
            lines=lines, labels=labels, model_forward_dict=model_forward_dict
        )
        accuracy_metrics = metric.get_metric()
        accuracy_metrics = accuracy_metrics["seq_label"]

        expected_precision = expected["expected_precision"]
        expected_recall = expected["expected_recall"]
        expected_fmeasure = expected["expected_fscore"]
        expected_micro_precision = expected["expected_micro_precision"]
        expected_micro_recall = expected["expected_micro_recall"]
        expected_micro_fscore = expected["expected_micro_fscore"]
        expected_macro_precision = expected["expected_macro_precision"]
        expected_macro_recall = expected["expected_macro_recall"]
        expected_macro_fscore = expected["expected_macro_fscore"]

        precision = accuracy_metrics["precision"]
        recall = accuracy_metrics["recall"]
        fscore = accuracy_metrics["fscore"]
        micro_precision = accuracy_metrics["micro_precision"]
        micro_recall = accuracy_metrics["micro_recall"]
        micro_fscore = accuracy_metrics["micro_fscore"]
        macro_precision = accuracy_metrics["macro_precision"]
        macro_recall = accuracy_metrics["macro_recall"]
        macro_fscore = accuracy_metrics["macro_fscore"]

        for class_label, precision_value in precision.items():
            assert precision_value == expected_precision[class_label]

        for class_label, recall_value in recall.items():
            assert recall_value == expected_recall[class_label]

        for class_label, fscore_value in fscore.items():
            assert fscore_value == expected_fmeasure[class_label]

        assert micro_precision == expected_micro_precision
        assert micro_recall == expected_micro_recall
        assert micro_fscore == expected_micro_fscore
        assert macro_precision == expected_macro_precision
        assert macro_recall == expected_macro_recall
        assert macro_fscore == expected_macro_fscore

    @pytest.mark.parametrize("report_type", ["wasabi"])
    def test_report_metric_works(self, setup_basecase, report_type):
        metric, lines, labels, model_forward_dict, expected = setup_basecase
        try:
            metric.report_metrics(report_type=report_type)
        except:
            pytest.fail(f"report_metric(report_type={report_type}) failed")

    def test_confusion_mtrx_works(self, setup_basecase):
        metric, lines, labels, model_forward_dict, expected = setup_basecase
        try:
            true_tag_indices = [[5, 4]]
            predicted_tag_indices = model_forward_dict["predicted_tags_seq_label"]
            metric.print_confusion_metrics(
                true_tag_indices=true_tag_indices,
                predicted_tag_indices=predicted_tag_indices,
                labels_mask=None,
            )
        except:
            pytest.fail("print_counfusion_metric() failed")

    def test_token_cls_accuracy_in_class_nursery(self):
        assert ClassNursery.class_nursery.get("TokenClassificationAccuracy") is not None
