from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="loan_strategy_analyzer",
    version="1.0.0",
    author="Sameer Gupta",
    author_email="your.email@example.com",
    description="AI-powered loan strategy analyzer for marketing optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/loan-strategy-analyzer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.21.0",
        "openai>=1.3.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.4.0",
        "scikit-learn>=1.3.0",
        "plotly>=5.17.0",
        "python-dateutil>=2.8.2",
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.9.0',
            'flake8>=6.1.0',
            'mypy>=1.5.0',
            'isort>=5.12.0',
            'jupyter>=1.0.0',
        ],
        'docs': [
            'sphinx>=7.1.0',
            'sphinx-rtd-theme>=1.3.0',
            'sphinx-autodoc-typehints>=1.24.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'loan-analyzer=loan_strategy_analyzer.cli:main',
        ],
    },
    package_data={
        'loan_strategy_analyzer': [
            'config/*.yaml',
            'data/*.csv',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    project_urls={
        # "Bug Tracker": "https://github.com/yourusername/loan-strategy-analyzer/issues",
        # "Documentation": "https://loan-strategy-analyzer.readthedocs.io/",
        # "Source Code": "https://github.com/yourusername/loan-strategy-analyzer",
    },
)