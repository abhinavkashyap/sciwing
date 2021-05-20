from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="sciwing",
    version="0.1.post8",
    packages=find_packages(exclude=("tests",)),
    url="https://github.com/abhinavkashyap/sciwing",
    license="",
    author="abhinav",
    author_email="abhinav@comp.nus.edu.sg",
    description="Modern Scientific Document Processing Framework ",
    entry_points={"console_scripts": ["sciwing=sciwing.commands.sciwing_group:main"]},
    long_description=README,
    long_description_content_type="text/markdown",
    install_requires=open("requirements.txt").readlines(),
)
