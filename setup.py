from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="sciwing",
    version="0.1.0b1",
    packages=find_packages(exclude=("tests",)),
    url="",
    license="",
    author="abhinav",
    author_email="abhinav@comp.nus.edu.sg",
    description="Modern ParseSect and ParseHed projects from WING-NUS",
    entry_points={"console_scripts": ["sciwing=sciwing.commands.sciwing_group:main"]},
    long_description=README,
    long_description_content_type="text/markdown",
)
