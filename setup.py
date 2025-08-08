from setuptools import setup, find_packages

setup(
    name="proteus",
    version="0.1.0",
    description="Computación sin Neuronas - Inteligencia desde Dinámicas Topológicas",
    author="Proteus Research Team",
    author_email="proteus@research.org",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "plotly>=5.14.0",
        "imageio>=2.31.0",
        "pillow>=10.0.0",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)