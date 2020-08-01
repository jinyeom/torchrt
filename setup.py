import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchrt",
    version="0.1.0",
    author="Jin Yeom",
    author_email="jinyeom95@gmail.com",
    description="A lightweight PyTorch >> ONNX >> TensorRT converter with a focus on simplicity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jinyeom/torchrt",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
