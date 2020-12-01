import click
from sciwing.models.neural_parscit import NeuralParscit
from sciwing.models.citation_intent_clf import CitationIntentClassification
from sciwing.models.generic_sect import GenericSect
from sciwing.models.sectlabel import SectLabel
from sciwing.models.i2b2 import I2B2NER


@click.command()
@click.argument("model")
def interact(model):
    """ Interact with pretrained models using command line.
    MODEL can be either of neural-parscit, citation-intent-clf, generic-sect, sect-label, i2b2-ner
    """
    pretrained_model = None
    if model == "neural-parscit":
        pretrained_model = NeuralParscit()
    elif model == "citation-intent-clf":
        pretrained_model = CitationIntentClassification()
    elif model == "generic-sect":
        pretrained_model = GenericSect()
    elif model == "sect-label":
        pretrained_model = SectLabel()
    elif model == "i2b2-ner":
        pretrained_model = I2B2NER()
    else:
        print(f"check --help for valid options")
        exit(1)
    pretrained_model.interact()
