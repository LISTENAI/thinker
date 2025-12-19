pip uninstall thinker -y
rm -rf dist/*
python setup.py sdist

pip install dist/thinker*.tar.gz
