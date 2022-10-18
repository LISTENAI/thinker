from ....enum_defines import Layout


def ProdShape(Shape):
    result = 1
    for x in Shape:
        result = result * x
    return result


def gemm_workspace_desc(
    matA, is_a_trans, is_a_pack, matB, is_b_trans, is_b_pack, dtype_size
):
    M = 0
    N = 0
    K = 0
    if False == is_a_trans and False == is_b_trans:
        M = matA[0]
        K = matA[1]
        N = matB[1]
    elif False == is_a_trans and True == is_b_trans:
        M = matA[0]
        K = matA[1]
        N = matB[0]
    elif True == is_a_trans and False == is_b_trans:
        M = matA[1]
        K = matA[0]
        N = matB[1]
    elif True == is_a_trans and True == is_b_trans:
        M = matA[1]
        K = matA[0]
        N = matB[0]

    workspace_bytes = 0
    if False == is_a_pack and M > 1:
        workspace_bytes += dtype_size * M * K
    if False == is_b_pack and N > 1:
        workspace_bytes += dtype_size * K * N
    return workspace_bytes


def get_deconv_workspace_desc(calc_kernel, X, W, attrs, outputs):
    group = attrs["group"]
    layout = attrs["layout"]
    input_shape = X.shape
    weight_shape = W.shape
    dtype_size = X.dtype.itemsize

    M = 0
    N = 0
    K = 0
    workspace_size = 0
    matA = []
    matB = []
    is_a_trans = 0
    is_b_trans = 0
    if layout == Layout.NCHW:
        M = ProdShape(weight_shape[1:])
        N = ProdShape(input_shape[2:])
        K = weight_shape[0] / group
        is_a_trans = 1
        matA = [K, M]
        is_b_trans = 0
        matB = [K, N]
    else:
        M = ProdShape(input_shape[1:3])
        N = ProdShape(weight_shape[1:])
        K = weight_shape[0] / group
        is_a_trans = 0
        matA = [M, K]
        is_b_trans = 0
        matB = [K, N]
        if group > 1:
            workspace_size += dtype_size * M * K

    workspace_size += dtype_size * M * N
    workspace_size += gemm_workspace_desc(
        matA, is_a_trans, False, matB, is_b_trans, False, dtype_size
    )
    return int(workspace_size)


def get_gemm_workspace_desc(calc_kernel, inputs, attrs, outputs):
    matA = inputs[0].shape
    matB = inputs[1].shape
    dtype_size = inputs[0].dtype.itemsize

    is_a_trans = attrs["transA"]
    is_b_trans = attrs["transB"]

    workspace_size = gemm_workspace_desc(
        matA, is_a_trans, False, matB, is_b_trans, False, dtype_size
    )
    return workspace_size


__all__ = [
    "gemm_workspace_desc",
    "get_deconv_workspace_desc",
    "get_gemm_workspace_desc",
]
