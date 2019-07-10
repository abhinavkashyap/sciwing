import pytest
import torch
from parsect.metrics.token_cls_accuracy import TokenClassificationAccuracy


@pytest.fixture
def setup_basecase():
    predicted_tags = [[1, 0]]
    labels = torch.LongTensor([[1, 0]])
    idx2labelname_mapping = {0: "good class", 1: "bad class"}

    expected_precision = {0: 1.0, 1: 1.0}
    expected_recall = {0: 1.0, 1: 1.0}
    expected_fmeasure = {0: 1.0, 1: 1.0}
    expected_macro_precision = 1.0
    expected_macro_recall = 1.0
    expected_macro_fscore = 1.0
    expected_num_tps = {0: 1.0, 1: 1.0}
    expected_num_fps = {0: 0.0, 1: 0.0}
    expected_num_fns = {0: 0.0, 1: 0.0}
    expected_micro_precision = 1.0
    expected_micro_recall = 1.0
    expected_micro_fscore = 1.0

    token_cls_metric = TokenClassificationAccuracy(
        idx2labelname_mapping=idx2labelname_mapping
    )

    iter_dict = {"label": labels}
    model_forward_dict = {"predicted_tags": predicted_tags}

    return (
        token_cls_metric,
        iter_dict,
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
        metric, iter_dict, model_forward_dict, expected = setup_basecase
        metric.calc_metric(iter_dict=iter_dict, model_forward_dict=model_forward_dict)
        accuracy_metrics = metric.get_metric()

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

    @pytest.mark.parametrize("report_type", ["wasabi", "paper"])
    def test_report_metric_works(self, setup_basecase, report_type):
        metric, iter_dict, model_forward_dict, expected = setup_basecase
        try:
            metric.report_metrics(report_type=report_type)
        except:
            pytest.fail(f"report_metric(report_type={report_type}) failed")

    def test_confusion_mtrx_works(self, setup_basecase):
        metric, iter_dict, model_forward_dict, expected = setup_basecase
        try:
            true_tag_indices = iter_dict["label"].tolist()
            predicted_tag_indices = model_forward_dict["predicted_tags"]
            metric.print_confusion_metrics(
                true_tag_indices=true_tag_indices,
                predicted_tag_indices=predicted_tag_indices,
            )
        except:
            pytest.fail("print_counfusion_metric() failed")
