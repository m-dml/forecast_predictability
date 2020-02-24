import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="predictability_utils",
    version="0.0.1",
    author="Marcel Nonnenmacher",
    author_email="marcel.nonnenmacher@hzg.de",
    description="Simple algorithms for seasonal weather predictions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mnonnenm/seasonal_forcast_predictability",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)