from setuptools import setup, find_packages

setup(
    name="transformer-decoder",
    version="0.1.0",
    description="A simple character-level Transformer decoder implementation",
    author="Your Name",
    author_email="you@example.com",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10",
        "numpy",
    ],
    python_requires=">=3.7",
)