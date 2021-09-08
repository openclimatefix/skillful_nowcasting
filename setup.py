from setuptools import setup, find_packages
import os

install_folder = os.path.dirname(os.path.realpath(__file__))
requirementPath = install_folder + '/requirements.txt'
with open(requirementPath) as f:
    install_requires = f.read().splitlines()

setup(
    name="skillful_nowcasting",
    version="0.0.4",
    packages=find_packages(),
    url="https://github.com/openclimatefix/skillful_nowcasting",
    license="MIT License",
    company="Open Climate Fix Ltd",
    author="Jacob Bieker",
    author_email="jacob@openclimatefix.org",
    description="PyTorch Skillful Nowcasting GAN Implementation",
    keywords=[
        "artificial intelligence",
        "deep learning",
        "transformer",
        "attention mechanism",
        "metnet",
        "forecasting",
        "remote-sensing",
        "gan"
    ],
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
