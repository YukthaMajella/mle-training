from setuptools import find_packages, setup

setup(
    name="HousePricingPredictor",  # Name of the package
    version="0.0.1",  # Version of the package
    packages=find_packages(
        where="src"
    ),  # Packages to include by automatically discovering them
    package_dir={"": "src"},  # Tells setuptools where the package code is located
    author="Majella Yuktha Biju",
    author_email="majella.biju@tigeranalytics.com",
    description="This package is used for the house pricing prediction problem.",
    long_description=open("README").read(),  # Long description from your README file
    long_description_content_type="text/markdown",  # Format of the long description
    url="https://github.com/YukthaMajella/mle-training",  # URL for the project
    python_requires=">=3.10",  # Minimum Python version supported
    include_package_data=True,  # Include non-Python files (e.g., README, LICENSE)
)
