import pytest
from falcon import testing
from sciwing.api.api import api
import falcon
import json
import pathlib
import sciwing.constants as constants

PATHS = constants.PATHS
OUTPUT_DIR = PATHS["OUTPUT_DIR"]


@pytest.fixture()
def client():
    return testing.TestClient(api)


@pytest.mark.skipif(
    not pathlib.Path(OUTPUT_DIR, "lstm_crf_science_ie_debug").exists(),
    reason="debug moel for lstm crf parscit does not exist",
)
class TestScienceIETaggerResource:
    def test_science_tagger_returns_okay(self, client):
        response = client.simulate_get(f"/science_ie_tagger?text={'single line'}")
        assert response.status == falcon.HTTP_200

    @pytest.mark.parametrize(
        "citation",
        ["first string", "string string string string", " ".join(["string"] * 10)],
    )
    def test_parscit_tagger_returns_same_len_labels(self, client, citation):
        response = client.simulate_get(f"/parscit_tagger?citation={citation}")
        response_content = response.content
        response_content = json.loads(response_content)
        response_content = response_content["label"]
        labels = response_content.split()
        assert len(labels) == len(citation.split())
