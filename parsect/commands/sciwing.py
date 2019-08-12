"""
The entry point to all the commands in the SciWING project
"""
import click
from parsect.commands.new import new


@click.group()
def sciwing():
    pass


if __name__ == "__main__":
    sciwing.add_command(new)
    sciwing()
