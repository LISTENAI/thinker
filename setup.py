import os
from setuptools import setup,find_packages

this_package_name="pythinker"
    
setup(
	name=this_package_name,
	version="0.1.0",
	description="A DeepLearning inference framework for venus",
	author="leofang3",
	author_email="leifang202209@163.com",
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

