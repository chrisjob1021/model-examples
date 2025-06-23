#!/usr/bin/env python3
"""Setup script for model-examples project"""

from setuptools import setup, find_packages

setup(
    name="model-examples",
    version="1.0.0",
    description="Example implementations of various neural network models",
    author="Model Examples Team",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
        "numpy>=1.21.0",
        "Pillow>=8.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
) 