from sciwing.data.seq_label import SeqLabel
import pytest


@pytest.fixture
def setup_seq_labels():
    labels = ["B-PER", "L-PER"]
    label = SeqLabel(labels=labels)
    return label


class TestSeqLabel:
    def test_labels_set(self, setup_seq_labels):
        label = setup_seq_labels
        tokens = label.tokens["seq_label"]
        for token in tokens:
            assert len(token.text) > 1
            assert isinstance(token.text, str)
