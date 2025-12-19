set -e
pushd ./
rm -rf build && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE="Debug"    \
    -DARCH="x86_64"                 \
    -DTHINKER_SHARED_LIB=ON         \
    -DTHINKER_PROFILE=OFF           \
    -DTHINKER_RESULT_DUMP=ON       \
    -DTHINKER_RESULT_CRC_PRINT=OFF  \
    -DTHINKER_RESOUCR_CRC_CHECK=OFF \
    -DTHINKER_USE_VENUS=OFF         \
    -DTHINKER_USE_ARCS=OFF          \
    -DTHINKER_USE_VENUSA=ON         \
    -DTHINKER_USE_MOSS=OFF          \
    -DTHINKER_CHECK_PLATFORM=ON     \
    ..

# make VERBOSE=1 -j16
make -j16
popd
