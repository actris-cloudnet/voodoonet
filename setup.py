from setuptools import find_packages, setup

version: dict = {}
with open("voodoonet/version.py", encoding="utf8") as f:
    exec(f.read(), version)  # pylint: disable=W0122

with open("README.md", encoding="utf8") as f:
    readme = f.read()

setup(
    name="voodoonet",
    version=version["__version__"],
    description="Voodoo method",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Willi Schimmel",
    author_email="willi.schimmel@uni-leipzig.de",
    url="https://github.com/actris-cloudnet/voodoonet",
    license="MIT License",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    setup_requires=["wheel"],
    install_requires=[
        "numpy",
        "scipy",
        "tqdm",
        "rpgpy>=0.12.1",
    ],
    extras_require={
        "dev": [
            "pre-commit",
            "pytest",
            "pytest-flakefinder",
            "pylint",
            "mypy",
            "types-tqdm",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
)
