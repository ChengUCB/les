from setuptools import setup, find_packages

setup(
    name="les",
    version="0.2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
    ],
    entry_points={
        "console_scripts": [
            "les=les.main:main",  # Adjust this if your entry point is different
        ],
    },
    author="Bingqing Cheng",
    description="Setup for LES",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ChengUCB/les",
    classifiers=[],
    python_requires=">=3.6",
)
