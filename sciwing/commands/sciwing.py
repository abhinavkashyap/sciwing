"""
The entry point to all the commands in the SciWING project
"""
import click
from sciwing.commands.new import new
from sciwing.commands.run import run
from sciwing.commands.test import test


@click.group()
def sciwing():
    pass


if __name__ == "__main__":
    sciwing.add_command(new)
    sciwing.add_command(run)
    sciwing.add_command(test)
    sciwing()
