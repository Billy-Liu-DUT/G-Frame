"""Setuptools compatibility shim for older system pip installations.

The canonical metadata remains in ``pyproject.toml``. This explicit shim keeps
wheel builds deterministic on the v1 host image, whose system setuptools is
too old to reliably consume PEP 621 metadata through build isolation.
"""

from setuptools import find_packages, setup


setup(
    name="g-frame-omnichem",
    version="0.2.0",
    description="G-Frame v2 chemistry data synthesis and full-parameter SFT orchestration.",
    long_description=open("readme.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"g_frame": ["prompt_assets/*.json"]},
    entry_points={"console_scripts": ["gframe=g_frame.cli:main"]},
)
