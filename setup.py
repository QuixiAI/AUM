# AUM-Ø — pure-Python package (Triton-only kernels; no CUDA/C++ build).
import ast
import os
import re
from pathlib import Path

from setuptools import find_packages, setup

this_dir = os.path.dirname(os.path.abspath(__file__))
PACKAGE_NAME = "aum_ssm"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def get_package_version():
    with open(Path(this_dir) / PACKAGE_NAME / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("AUM_LOCAL_VERSION")
    return f"{public_version}+{local_version}" if local_version else str(public_version)


setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(exclude=("build", "tests", "dist", "docs", "aum_ssm.egg-info")),
    author="Eric Hartford",
    description="AUM-Ø: Attentive Unfolding Modulation with Silence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "triton>=3.5.0",
        "einops",
        "transformers",
        "packaging",
        # optional: "causal-conv1d>=1.4.0"  (short causal conv in the U phase)
    ],
)
