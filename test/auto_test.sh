set -e

test_dir=./model.test
function download_onnx_thinker_models()
{
    set +e
    dir_name=$1
    file1_name=$2
    file2_name=$3
    file3_name=$4
    mkdir -p "$test_dir/${dir_name}"
    cp  /data/user/thinker/models/${dir_name}/${file1_name} $test_dir/${dir_name}/${file1_name}
    cp /data/user/thinker/models/${dir_name}/${file2_name} $test_dir/${dir_name}/${file2_name}
    cp /data/user/thinker/models/${dir_name}/${file3_name} $test_dir/${dir_name}/${file3_name}
}

#################### fetch onnx graph and config file ####################
export LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH
# module load gcc/5.4.0-os7.2
if [ -d "$test_dir" ]; then
    rm -rf $test_dir
fi

mkdir -p $test_dir
pushd ./
download_onnx_thinker_models "test_conv1d" "conv1d.onnx" "input.bin" "output.bin"
download_onnx_thinker_models "test_conv2d" "net.onnx" "input.bin" "output.bin"
download_onnx_thinker_models "test_gru" "gru_int.onnx" "input.bin" "output.bin"
download_onnx_thinker_models "test_batchnorm" "batchnormInt.onnx" "input.bin" "output.bin"
download_onnx_thinker_models "test_softmaxint" "softmaxint.onnx" "input.bin" "output.bin"
download_onnx_thinker_models "test_logsoftmax" "logsoftmaxint.onnx" "input.bin" "output.bin"
download_onnx_thinker_models "test_iqsigmoid" "iqsigmoid.onnx" "input.bin" "output.bin"
download_onnx_thinker_models "test_layernorm" "layernorm_int.onnx" "input.bin" "output.bin"
download_onnx_thinker_models "test_shufflechannel" "shuffle_net_10_09.onnx" "input.bin" "output.bin"
popd

###################### compile thinker.so ######################
CMAKE_ROOT=/home/bitbrain/bzcai/anaconda3/bin
pushd ./
pwdn
rm -rf lib
rm -rf bin
rm -rf build
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE="Release"  \
    -DTHINKER_SHARED_LIB=ON         \
    -DTHINKER_PROFILE=OFF           \
    -DTHINKER_DUMP=OFF              \
    ..
make -j4
popd

pushd ./
rm -rf dist/*
python setup.py sdist

pip install dist/pythinker*.tar.gz
popd

###################### compile test.cpp ######################
pushd ./
tpacker -g $test_dir/test_conv2d/net.onnx -s Remove_QuantDequant -o $test_dir/test_conv2d/model.bin
tpacker -g $test_dir/test_conv1d/conv1d.onnx -s Remove_QuantDequant -o $test_dir/test_conv1d/model.bin
tpacker -g $test_dir/test_batchnorm/batchnormInt.onnx -s Remove_QuantDequant -o $test_dir/test_batchnorm/model.bin
tpacker -g $test_dir/test_softmaxint/softmaxint.onnx -s Remove_QuantDequant -o $test_dir/test_softmaxint/model.bin
tpacker -g $test_dir/test_logsoftmax/logsoftmaxint.onnx -s Remove_QuantDequant -o $test_dir/test_logsoftmax/model.bin
tpacker -g $test_dir/test_iqsigmoid/iqsigmoid.onnx -s Remove_QuantDequant -o $test_dir/test_iqsigmoid/model.bin
tpacker -g $test_dir/test_layernorm/layernorm_int.onnx -s Remove_QuantDequant -o $test_dir/test_layernorm/model.bin
tpacker -g $test_dir/test_shufflechannel/shuffle_net_10_09.onnx -s Remove_QuantDequant -o $test_dir/test_shufflechannel/model.bin
tpacker -g $test_dir/test_gru/gru_int.onnx -s Remove_QuantDequant -o $test_dir/test_gru/model.bin

cd test/linux_x86   && rm -rf build && rm -rf bin && mkdir -p build && cd build && cmake  -DCMAKE_BUILD_TYPE="Release" ../
make
popd

######################## run the engine ######################
export LD_LIBRARY_PATH=./bin/:$LD_LIBRARY_PATH
./test/linux_x86/bin/test_x86
rm -rf $test_dir