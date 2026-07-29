from setuptools import setup, find_packages

setup(
    name="image-dehazing",
    version="1.0.0",
    author="Senior CV & AI Research Team",
    author_email="contact@dehazeai.org",
    description="AI-Based Single Image Dehazing System using Transformer-Based Deep Learning",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/image-dehazing",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.9",
)
