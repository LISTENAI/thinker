set -e
pushd ./
rm -rf build && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE="Debug"      \
    -DTHINKER_SHARED_LIB=ON            \
    -DTHINKER_PROFILE=OFF              \
    -DTHINKER_RESULT_DUMP=ON           \
    -DDTHINKER_RESULT_CRC_PRINT=OFF    \
    -DDTHINKER_RESOUCR_CRC_CHECK=OFF   \
    -DDTHINKER_TARGET_PLATFORM="ARCS"  \
    -DDTHINKER_TARGET_CHECK=ON         \
    -DTHINKER_USE_MOSS=OFF             \
    -DTHINKER_USE_MTQ=OFF              \
    -DTHINKER_USE_NNBLAS=OFF           \
    ..

# make VERBOSE=1 -j16
make -j16
popd
