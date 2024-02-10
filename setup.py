import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rippy",                     # This is the name of the package
    version="0.0.5",                        # The initial release version
    author="James Norman",                     # Full name of the author
    description="Reinsurance Pricing in Python",
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(where='rippy'),    # List of all python modules to be installed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.10',                # Minimum version requirement of the package
    py_modules=["rippy"],             # Name of the python package
    package_dir={'':'.'},     # Directory of the source code of the package
    install_requires=['numpy','scipy'] ,                    # Install other dependencies if any
    project_urls={  # Optional
        "Bug Reports": "https://github.com/ProteusLLP/rippy/issues",
        "Source": "https://github.com/ProteusLLP/rippy",
    
    },
    extras_require={"gpu":["cupy-cuda11x"]},
    keywords='reinsurance, actuarial, insurance',
)