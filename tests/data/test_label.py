import pytest
from sciwing.data.label import Label


class TestLabel:
    def test_label_str_getter(self):
        label = Label(label_str="Introduction")
        assert label.label_str == "Introduction"
