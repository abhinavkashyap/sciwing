import click
from sciwing.commands.new import new
from sciwing.commands.run import run
from sciwing.commands.test import test


@click.group()
def sciwing():
    """Root command for everything else in sciwing

    Returns
    -------
    None

    """
    pass


if __name__ == "__main__":
    sciwing.add_command(new)
    sciwing.add_command(run)
    sciwing.add_command(test)
    sciwing()
