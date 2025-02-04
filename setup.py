from setuptools import setup, find_packages

setup(
    name="deep-derivatives",
    version="0.1.0",
    description="Deep learning framework for derivative pricing and hedging",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Kheri Hughes",
    url="https://github.com/syphinx/deep-derivatives",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "torch>=1.9.0",
        "gym>=0.21.0",
        "matplotlib>=3.4.3",
        "seaborn>=0.11.2",
        "tqdm>=4.62.0",
        "python-dotenv>=0.19.0",
        "pytest>=6.2.5",
        "pytest-cov>=2.12.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="finance, derivatives, deep-learning, reinforcement-learning, option-pricing"
) 