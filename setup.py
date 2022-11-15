import os
from setuptools import setup,find_packages

this_package_name="pythinker"
    
setup(
	name=this_package_name,
	version="1.0.0",
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
          "console_scripts": ["tpacker=thinker.tpacker:main"]
    },

    classifiers=["Programming Language :: Python :: 3",
                    "License :: OSI Approved :: Apache Software License",
                    "Operating System :: OS Independent",],
    python_requires='>=3.6'
)

