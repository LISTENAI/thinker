#!/bin/sh
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

BUILD_DIR=${BUILD_DIR:-"$ROOT_DIR/build"}
BUILD_TYPE=${BUILD_TYPE:-Debug}
BUILD_JOBS=${BUILD_JOBS:-16}
TARGET_PLATFORM=${THINKER_TARGET_PLATFORM:-VENUSA}
THINKER_PARAM_CHECK=${THINKER_PARAM_CHECK:-OFF}
THINKER_RUNTIME_CHECK=${THINKER_RUNTIME_CHECK:-OFF}
USE_MOSS=${THINKER_USE_MOSS:-OFF}
MOSS_RES_DIR=${MOSS_RES_DIR:-"$ROOT_DIR/moss_res"}
MOSS_MODELS=${MOSS_MODELS:-"anyreid;face_keypoint"}

moss_model_list() {
    kind=$1
    result=
    old_ifs=$IFS
    IFS=';'
    for model in $MOSS_MODELS; do
        [ -n "$model" ] || continue
        case "$kind" in
            host) item="$MOSS_RES_DIR/$model/${model}_host.c" ;;
            getter) item="mGetModel_$model" ;;
            name) item="$model" ;;
            *) item= ;;
        esac
        if [ -n "$result" ]; then
            result="$result;$item"
        else
            result="$item"
        fi
    done
    IFS=$old_ifs
    printf '%s' "$result"
}

rm -rf "$BUILD_DIR"

set -- \
    -S "$ROOT_DIR" \
    -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DTHINKER_SHARED_LIB=ON \
    -DTHINKER_PROFILE=OFF \
    -DTHINKER_RESULT_DUMP=ON \
    -DTHINKER_RESULT_CRC_PRINT=OFF \
    -DTHINKER_RESOUCR_CRC_CHECK=OFF \
    -DTHINKER_TARGET_PLATFORM="$TARGET_PLATFORM" \
    -DTHINKER_TARGET_CHECK=ON \
    -DTHINKER_PARAM_CHECK="$THINKER_PARAM_CHECK" \
    -DTHINKER_RUNTIME_CHECK="$THINKER_RUNTIME_CHECK"

case "$USE_MOSS" in
    ON|on|1|TRUE|true|YES|yes)
        MOSS_HOST_SOURCES=${THINKER_MOSS_HOST_SOURCES:-$(moss_model_list host)}
        MOSS_MODEL_GETTERS=${THINKER_MOSS_MODEL_GETTERS:-$(moss_model_list getter)}
        MOSS_MODEL_NAMES=${THINKER_MOSS_MODEL_NAMES:-$(moss_model_list name)}
        set -- "$@" \
            -DTHINKER_USE_MOSS=ON \
            -DTHINKER_USE_NNBLAS=OFF \
            -DTHINKER_MOSS_HOST_SOURCES="$MOSS_HOST_SOURCES" \
            -DTHINKER_MOSS_MODEL_GETTERS="$MOSS_MODEL_GETTERS" \
            -DTHINKER_MOSS_MODEL_NAMES="$MOSS_MODEL_NAMES"
        ;;
    *)
        set -- "$@" -DTHINKER_USE_MOSS=OFF
        ;;
esac

cmake "$@"
cmake --build "$BUILD_DIR" -j "$BUILD_JOBS"
