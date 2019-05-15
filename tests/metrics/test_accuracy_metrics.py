import pytest
import torch
from parsect.metrics.accuracy_metrics import Accuracy


@pytest.fixture
def setup_data_basecase():
    predicted_probs = torch.FloatTensor([[0.1, 0.9],
                                         [0.7, 0.3]])
    labels = torch.LongTensor([1, 0])

    expected_precision = {0: 1.0, 1: 1.0}
    expected_recall = {0: 1.0, 1: 1.0}
    expected_fmeasure = {0: 1.0, 1: 1.0}

    accuracy = Accuracy()
    return predicted_probs, labels, accuracy, {
        'expected_precision': expected_precision,
        'expected_recall': expected_recall,
        'expected_fscore': expected_fmeasure
    }


@pytest.fixture
def setup_data_one_true_class_missing():
    """
    The batch of instances during training might not have all
    true classes. What happens in that case??
    The test case here captures the situation
    :return:
    """
    predicted_probs = torch.FloatTensor([[0.8, 0.1, 0.2],
                                         [0.2, 0.5, 0.3]])
    labels = torch.LongTensor([0, 2])

    expected_precision = {0: 1.0, 1: 0.0, 2: 0.0}
    expected_recall = {0: 1.0, 1: 0.0, 2: 0.0}
    expected_fscore = {0: 1.0, 1: 0.0, 2: 0.0}

    accuracy = Accuracy()

    return predicted_probs, labels, accuracy, {
        'expected_precision': expected_precision,
        'expected_recall': expected_recall,
        'expected_fscore': expected_fscore
    }


@pytest.fixture
def setup_data_to_test_length():
    predicted_probs = torch.FloatTensor([[0.1, 0.8, 0.2],
                                         [0.2, 0.3, 0.5]])
    labels = torch.LongTensor([0, 2])

    accuracy = Accuracy()

    expected_length = 3

    return predicted_probs, labels, accuracy, expected_length


class TestAccuracy:
    def test_accuracy_basecase(self, setup_data_basecase):
        predicted_probs, labels, accuracy, expected = setup_data_basecase
        expected_precision = expected['expected_precision']
        expected_recall = expected['expected_recall']
        expected_fmeasure = expected['expected_fscore']

        accuracy_metrics = accuracy.get_accuracy(predicted_probs, labels)

        precision = accuracy_metrics['precision']
        recall = accuracy_metrics['recall']
        fscore = accuracy_metrics['fscore']

        for class_label, precision_value in precision.items():
            assert precision_value == expected_precision[class_label]

        for class_label, recall_value in recall.items():
            assert recall_value == expected_recall[class_label]

        for class_label, fscore_value in fscore.items():
            assert fscore_value == expected_fmeasure[class_label]

    def test_accuracy_one_true_class_missing(self, setup_data_one_true_class_missing):
        predicted_probs, labels, accuracy, expected = setup_data_one_true_class_missing
        expected_precision = expected['expected_precision']
        expected_recall = expected['expected_recall']
        expected_fscore = expected['expected_fscore']

        accuracy_metrics = accuracy.get_accuracy(predicted_probs, labels)

        precision = accuracy_metrics['precision']
        recall = accuracy_metrics['recall']
        fscore = accuracy_metrics['fscore']

        for class_label, precision_value in precision.items():
            assert precision_value == expected_precision[class_label]

        for class_label, recall_value in recall.items():
            assert recall_value == expected_recall[class_label]

        for class_label, fscore_value in fscore.items():
            assert fscore_value == expected_fscore[class_label]

    def test_length(self, setup_data_to_test_length):
        predicted_probs, labels, accuracy, expected_length = setup_data_to_test_length

        metrics = accuracy.get_accuracy(predicted_probs, labels)
        precision = metrics['precision']

        assert len(precision.keys()) == expected_length

    def test_print_confusion_matrix_works(self, setup_data_basecase):
        predicted_probs, labels, accuracy, expected = setup_data_basecase
        accuracy.print_confusion_metrics(predicted_probs,
                                         labels)
