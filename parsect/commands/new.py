import click
from parsect.commands.new_dataset import create_new_dataset_interactive


@click.command()
@click.argument("entity_type")
def new(entity_type):
    if entity_type == "dataset":
        create_new_dataset_interactive()
    elif entity_type == "model":
        pass


if __name__ == "__main__":
    new()
