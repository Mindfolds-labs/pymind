from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pymind",
    version="0.2.0",
    author="Seu Nome",
    author_email="seu.email@example.com",
    description="PyMind - Rede Neural com Dendritos e Memória Engram",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seuusername/pymind",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
    ],
)
