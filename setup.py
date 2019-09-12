from setuptools import setup, find_packages

setup(
    name="sciwing",
    version="0.1.0b",
    packages=find_packages(exclude=("tests",)),
    url="",
    license="",
    author="abhinav",
    author_email="abhinav@comp.nus.edu.sg",
    description="Modern ParseSect and ParseHed projects from WING-NUS",
    entry_points={"console_scripts": ["sciwing=sciwing.commands.sciwing_group:main"]},
)
