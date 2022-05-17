'pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113'

import setuptools

setuptools.setup(
    name="nodeik",
    version="0.0.1",
    author="",
    author_email="",
    description="",
    url="",
    project_urls={
        "Documentation": "https://github.io/",
    },
    long_description="",
    long_description_content_type="text/markdown",
    license="MIT",
    packages=setuptools.find_packages(),
    package_data={
        "": []
        },
    classifiers=[
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "warp-lang", 'usd-core', 'urdfpy'],
    python_requires=">=3.7"
)