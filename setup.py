from setuptools import setup

setup(
    name="sciwing",
    version="0.0",
    packages=["sciwing"],
    url="",
    license="",
    author="abhinav",
    author_email="abhinav@comp.nus.edu.sg",
    description="Modern ParseSect and ParseHed projects from WING-NUS",
    entry_points={"console_scripts": ["sciwing=sciwing.commands.sciwing:main"]},
)
