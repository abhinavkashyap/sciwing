import pytest
from sciwing.data.label import Label


class TestLabel:
    def test_label_str_getter(self):
        label = Label(text="Introduction", namespace="label")
        tokens = label.tokens["label"]
        token_text = tokens[0].text
        assert token_text == "Introduction"
