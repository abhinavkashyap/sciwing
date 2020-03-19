import pytest
from sciwing.data.contextual_lines import LinesWithContext


@pytest.fixture
def line_context():
    line = "This is a single line"
    context = ["This is the first contextual line", "This is the second context"]
    return line, context


class TestLinesWithContext:
    def test_namespaces(self, line_context):
        line, context = line_context
        line_with_context = LinesWithContext(text=line, context=context)
        namespaces = line_with_context.namespaces
        assert namespaces == ["tokens"]

    def test_context(self, line_context):
        line, context = line_context
        line_with_context = LinesWithContext(text=line, context=context)
        context_tokens = line_with_context.get_context_tokens(namespace="tokens")
        assert len(context_tokens) == 2
