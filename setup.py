"""
Setup script for Cats vs Dogs MLOps Project
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cats-dogs-classifier",
    version="1.0.0",
    author="MLOps Team",
    author_email="mlops@example.com",
    description="MLOps project for Cats vs Dogs image classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mlops-cats-dogs",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "cats-dogs-train=src.models.train:main",
            "cats-dogs-evaluate=src.models.evaluate:main",
            "cats-dogs-serve=src.api.main:main",
        ],
    },
)
