set -e

function download_onnx_thinker_models()
{
    set +e
    dir_name=$1
    file1_name=$2
    file2_name=$3
    file3_name=$4
    mkdir -p "model/${dir_name}"
    cp  /data/user/thinker/models/${dir_name}/${file1_name} model/${dir_name}/${file1_name}
    cp /data/user/thinker/models/${dir_name}/${file2_name} model/${dir_name}/${file2_name}
    cp /data/user/thinker/models/${dir_name}/${file3_name} model/${dir_name}/${file3_name}

}

#################### fetch onnx graph and config file ####################
export LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH
module load gcc/5.4.0-os7.2
mkdir -p model
mkdir -p resource
pushd ./
download_onnx_thinker_models "test_conv2d" "net.onnx" "input.bin" "output.bin"
download_onnx_thinker_models "test_conv1d" "conv1d.onnx" "input.bin" "output.bin"
download_onnx_thinker_models "test_batchnorm" "batchnormInt.onnx" "input.bin" "output.bin"
download_onnx_thinker_models "test_softmaxint" "softmaxint.onnx" "input.bin" "output.bin"
download_onnx_thinker_models "test_logsoftmax" "logsoftmaxint.onnx" "input.bin" "output.bin"
download_onnx_thinker_models "test_iqsigmoid" "iqsigmoid.onnx" "input.bin" "output.bin"
download_onnx_thinker_models "test_layernorm" "layernorm_int.onnx" "input.bin" "output.bin"
download_onnx_thinker_models "test_shufflechannel" "shuffle_net_10_09.onnx" "input.bin" "output.bin"
download_onnx_thinker_models "test_gru" "gru_int.onnx" "input.bin" "output.bin"
popd

###################### compile thinker.so ######################
CMAKE_ROOT=/home/bitbrain/bzcai/anaconda3/bin
pushd ./
pwd
rm -rf lib
rm -rf bin
rm -rf build
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE="Release"  \
    -DTHINKER_SHARED_LIB=ON         \
    -DTHINKER_PROFILE=ON           \
    -DTHINKER_DUMP=ON              \
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
tpacker -g model/test_conv2d/net.onnx -s Remove_QuantDequant -o model/test_conv2d/model.bin
tpacker -g model/test_conv1d/conv1d.onnx -s Remove_QuantDequant -o model/test_conv1d/model.bin
tpacker -g model/test_batchnorm/batchnormInt.onnx -s Remove_QuantDequant -o model/test_batchnorm/model.bin
tpacker -g model/test_softmaxint/softmaxint.onnx -s Remove_QuantDequant -o model/test_softmaxint/model.bin
tpacker -g model/test_logsoftmax/logsoftmaxint.onnx -s Remove_QuantDequant -o model/test_logsoftmax/model.bin
tpacker -g model/test_iqsigmoid/iqsigmoid.onnx -s Remove_QuantDequant -o model/test_iqsigmoid/model.bin
tpacker -g model/test_layernorm/layernorm_int.onnx -s Remove_QuantDequant -o model/test_layernorm/model.bin
tpacker -g model/test_shufflechannel/shuffle_net_10_09.onnx -s Remove_QuantDequant -o model/test_shufflechannel/model.bin
tpacker -g model/test_gru/gru_int.onnx -s Remove_QuantDequant -o model/test_gru/model.bin

cd test/linux_x86   && rm -rf build && rm -rf bin && mkdir -p build && cd build && cmake  -DCMAKE_BUILD_TYPE="Release" ../
make
popd

######################## run the engine ######################
export LD_LIBRARY_PATH=./bin/:$LD_LIBRARY_PATH
./test/linux_x86/bin/test_x86