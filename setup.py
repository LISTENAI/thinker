import os
from setuptools import setup,find_packages

this_package_name="thinker"
    
setup(
	name=this_package_name,
	version="0.1.0",
	description="A DeepLearning inference framework for venus",
	author="thinker",
	author_email="listener@iflytek.com",
	url="https://git-in.iflytek.com/RS_RDG_AI_Group/bitbrain/thinker",
	packages=find_packages(),
	include_package_data=True,
    install_requires=[
    'onnx'
    ],
    entry_points={
          "console_scripts": ["tpacker=thinker.tpacker:main"]
    },

    classifiers=[
        "Operating System :: OS Independent",
        "Intended Audience :: Developers and Researchers",
        "License :: OSI Approved :: iflytek internal License",
        "Programming Language :: python",
    ],

)

