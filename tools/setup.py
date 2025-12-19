import os
from setuptools import setup,find_packages

this_package_name="thinker"
    
setup(
	name=this_package_name,
	version="3.0.2",
	description="A DeepLearning inference framework for venus",
	author="thinker",
	author_email="leifang@listenai.com",
	url="https://github.com/LISTENAI/thinker",
	packages=find_packages(),
	include_package_data=True,
    install_requires=[
    'onnx'
    ],
    entry_points={
          "console_scripts": [
            "tpacker=tpacker.tpacker:main",
            "tvalidator=tvalidator.validator:main",
            "tprofile=tprofile.src.onnx_profile:main",

            ],
    },

    classifiers=[
        "Operating System :: OS Independent",
        "Intended Audience :: Developers and Researchers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
    ],

)

