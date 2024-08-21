from pathlib import Path
from setuptools import setup, find_packages

requirements = ["keras <= 2.15",
                "tensorflow",
                "scikit-learn",
                "numpy >=1.8",
                "pandas",
                "matplotlib", # >= 3.1.1",
                "astropy >=1.3",
                "scipy >=0.14",
                ]

data = {"pycadet": ["*.hdf5",]}

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

MAJOR = "0"
MINOR = "2"
MICRO = "0"
version = "%s.%s.%s" % (MAJOR, MINOR, MICRO)

setup(
    name="pycadet",
    version=version,
    author="Tomas Plsek",
    author_email="plsek@physics.muni.cz",
    description="Cavity Detection Tool",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/tomasplsek/CADET",
    install_requires=requirements,
    packages=find_packages(exclude=("docs", "training_testing", "examples",)),
    package_data=data,
    include_package_data=False,
)