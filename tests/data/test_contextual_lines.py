import pytest
from sciwing.data.contextual_lines import LineWithContext


@pytest.fixture
def line_context():
    line = "This is a single line"
    context = ["This is the first contextual line", "This is the second context"]
    return line, context


class TestLinesWithContext:
    def test_namespaces(self, line_context):
        line, context = line_context
        line_with_context = LineWithContext(text=line, context=context)
        namespaces = line_with_context.namespaces
        assert "tokens" in namespaces
        assert "contextual_tokens" in namespaces
