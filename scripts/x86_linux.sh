set -e
module load gcc/7.3.0-os7.2
CMAKE_ROOT=/home/bitbrain/bzcai/anaconda3/bin
pushd ./
rm -rf build && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE="Debug"   \
    -DARCH="x86_64"                \
    -DTHINKER_SHARED_LIB=ON        \
    -DTHINKER_PROFILE=OFF          \
    -DTHINKER_DUMP=OFF             \
    -DTHINKER_USE_VENUS=ON         \
    ..

# make VERBOSE=1 -j16
make -j16
popd

pushd ./
rm -rf dist/*
python setup.py sdist

pip install dist/thinker*.tar.gz
popd