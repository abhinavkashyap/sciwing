import click
from sciwing.commands.new_dataset import create_new_dataset_interactive


@click.command()
@click.argument("entity_type")
def new(entity_type):
    """ Sub-Command to create new models and datasets

    Parameters
    ----------
    entity_type : str
        indicates the kind of new entity that is created

    """
    if entity_type == "dataset":
        create_new_dataset_interactive()
    elif entity_type == "model":
        pass


if __name__ == "__main__":
    new()
