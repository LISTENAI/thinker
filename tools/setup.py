import os
from setuptools import setup,find_packages

this_package_name="pythinker"
    
setup(
	name=this_package_name,
	version="3.0.11",
	description="A DeepLearning inference framework for venus",
	author="listenai",
	author_email="lingerthinker@listenai.com",
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
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
    ],

)

