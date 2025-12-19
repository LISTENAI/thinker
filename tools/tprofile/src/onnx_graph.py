import os
import onnx
from onnx import helper, TensorProto
import numpy as np
import copy
from typing import List
from typing import Dict, List, Optional, Any, Tuple


class Node:
    """
    onnx node.
    """

    def __init__(
        self,
        name: str = "",
        op_type: str = "",
        input: tuple = (),
        output: tuple = (),
        attributes: dict = {},
    ) -> None:
        self.name = name
        self.op_type = op_type
        self.input = input
        self.output = output
        self.attributes = attributes

    def get_name(self) -> str:
        return name

    def get_op_type(self) -> str:
        return self.op_type

    def get_input(self) -> List[str]:
        return list(self.input)

    def get_output(self) -> List[str]:
        return list(self.output)

    def get_attributes(self) -> dict:
        return self.attributes

    def set_name(self, name: str) -> None:
        self.name = name

    def set_op_type(self, op_type: str) -> None:
        self.op_type = op_type

    def set_input(self, input: tuple) -> None:
        self.input = input

    def set_output(self, output: tuple) -> None:
        self.output = output

    def set_attributes(self, attributes: dict) -> None:
        self.attributes = attributes

    def __str__(self) -> str:
        return f"\tname:{self.name}, op_type:{self.op_type}, input:{self.input}, output:{self.output}, attribute:{self.attributes}"


class Tensor:
    """
    onnx tensor.
    """

    def __init__(
        self,
        name: str = "",
        dtype: np.dtype = None,
        shape: tuple = None,
        init_tensor: np.ndarray = None,
    ) -> None:
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.init_tensor = init_tensor
        self.__update__()

    def set_name(self, name: str) -> None:
        self.name = name

    def set_shape(self, shape: tuple) -> None:
        self.shape = shape

    def set_dtype(self, dtype: np.dtype) -> None:
        self.dtype = dtype

    def set_value(self, init_tensor: np.ndarray) -> None:
        self.init_tensor = init_tensor
        self.__update__()

    def get_name(self) -> str:
        return self.name

    def get_shape(self) -> tuple:
        return self.shape

    def get_dtype(self) -> np.dtype:
        return self.dtype

    def get_value(self) -> np.ndarray:
        return self.init_tensor

    def get_size(self) -> int:
        size = 1
        for dim in self.shape:
            size *= dim
        return size

    def get_bitsize(self) -> int:
        return self.dtype.itemsize * 8

    def __update__(self) -> None:
        if self.init_tensor is not None:
            self.shape = self.init_tensor.shape
            self.dtype = self.init_tensor.dtype

    @staticmethod
    def from_array(arr: np.ndarray, name: str = "") -> "Tensor":
        return Tensor(name, arr.dtype, arr.shape, arr)

    @staticmethod
    def from_type(dtype: np.dtype, shape: List[int], name: str = "") -> "Tensor":
        arr = np.zeros(shape, dtype=dtype)
        return Tensor(name, arr.dtype, arr.shape, arr)

    def __str__(self) -> str:
        if isinstance(self.init_tensor, np.ndarray):
            return f"\tname:{self.name}, shape:{self.init_tensor.shape}, dtype:{self.init_tensor.dtype}"
        else:
            return f"\tname:{self.name}, shape:{self.shape}, dtype:{self.dtype}"


class Graph:
    """
    onnx graph.
    """

    def __init__(self, onnx_path):
        """
        init variable
        """
        self.name = ""
        self.input = []
        self.output = []
        self.node = []
        self.value_info = []
        self.initializer = []
        self.constant = []
        self.otype2ntype = {
            1: np.dtype("float32"),
            2: np.dtype("uint8"),
            3: np.dtype("int8"),
            4: np.dtype("uint16"),
            5: np.dtype("int16"),
            6: np.dtype("int32"),
            7: np.dtype("int64"),
            8: np.dtype("bool"),
        }
        self.ntype2otype = {
            np.dtype("float32"): 1,
            np.dtype("uint8"): 2,
            np.dtype("int8"): 3,
            np.dtype("uint16"): 4,
            np.dtype("int16"): 5,
            np.dtype("int32"): 6,
            np.dtype("int64"): 7,
            np.dtype("bool"): 8,
        }
        self.parse_onnx(onnx_path)

    def parse_onnx(self, onnx_path):
        """
        parse onnx
        """
        onnx_model = onnx.load(onnx_path)
        graph = onnx_model.graph
        self.name = graph.name
        if graph.name == "":
            self.name = os.path.splitext(os.path.basename(onnx_path))[0]
        for input in graph.input:
            name = input.name
            if input.type.HasField("tensor_type"):
                tensor_type = input.type.tensor_type
                dtype = self.otype2ntype[tensor_type.elem_type]
                shape = []
                for dim in tensor_type.shape.dim:
                    if dim.dim_param != "":
                        shape.append(dim.dim_param)
                    else:
                        shape.append(dim.dim_value)
            self.input.append(Tensor(name, dtype, shape))
        for output in graph.output:
            name = output.name
            if output.type.HasField("tensor_type"):
                tensor_type = output.type.tensor_type
                dtype = self.otype2ntype[tensor_type.elem_type]
                shape = []
                for dim in tensor_type.shape.dim:
                    if dim.dim_param != "":
                        shape.append(dim.dim_param)
                    else:
                        shape.append(dim.dim_value)
            self.output.append(Tensor(name, dtype, shape))
        for value_info in graph.value_info:
            name = value_info.name
            tensor_type = value_info.type.tensor_type
            dtype = self.otype2ntype[tensor_type.elem_type]
            shape = []
            for dim in tensor_type.shape.dim:
                if dim.dim_param != "":
                    shape.append(dim.dim_param)  # dynamic dim
                else:
                    shape.append(dim.dim_value)
            self.value_info.append(Tensor(name, dtype, shape))
        for node in graph.node:
            attributes = {}
            if node.op_type == "Constant":
                assert len(node.attribute) == 1
            for attr in node.attribute:
                value = None
                for f in ["f", "i", "s"]:
                    if attr.HasField(f):
                        value = getattr(attr, f)
                        # Needed for supporting python version > 3.5
                        if isinstance(value, bytes):
                            value = value.decode(encoding="utf-8")
                for f in ["floats", "ints", "strings"]:
                    if list(getattr(attr, f)):
                        value = tuple(getattr(attr, f))
                for f in ["t"]:
                    if attr.HasField(f):
                        tensor_proto = attr.t
                        tensor_shape = list(tensor_proto.dims)
                        tensor_dtype = self.otype2ntype[tensor_proto.data_type]
                        value = getattr(attr, f)
                        value = np.frombuffer(value.raw_data, dtype=tensor_dtype)
                        value = value.reshape(tensor_shape)
                attributes[attr.name] = value
            self.node.append(
                Node(
                    node.name,
                    node.op_type,
                    list(node.input),
                    list(node.output),
                    attributes,
                )
            )
        for init in graph.initializer:
            # Basic types.
            assert (
                init.data_type in self.otype2ntype
            ), f"onnx type unsuport, type = {init.data_type}"
            init_tensor = np.frombuffer(
                init.raw_data, dtype=self.otype2ntype[init.data_type]
            )
            init_tensor = init_tensor.reshape(init.dims)
            self.initializer.append(Tensor(init.name, init_tensor=init_tensor.copy()))

    def save_onnx(self, onnx_path):
        """
        save onnx
        """
        inputs = []
        for tensor in self.input:
            inputs.append(
                helper.make_tensor_value_info(
                    tensor.name, self.ntype2otype[tensor.dtype], tensor.shape
                )
            )
        outputs = []
        for tensor in self.output:
            outputs.append(
                helper.make_tensor_value_info(
                    tensor.name, self.ntype2otype[tensor.dtype], tensor.shape
                )
            )
        value_infos = []
        for tensor in self.value_info:
            value_infos.append(
                helper.make_tensor_value_info(
                    tensor.name, self.ntype2otype[tensor.dtype], tensor.shape
                )
            )
        initializers = []
        for tensor in self.initializer:
            if hasattr(onnx, "numpy_helper"):  # fix onnx 1.7.0
                initializers.append(
                    onnx.numpy_helper.from_array(tensor.init_tensor, tensor.name)
                )
            else:
                initializers.append(self.from_array(tensor.init_tensor, tensor.name))
        # Create a node (NodeProto) - This is based on Pad-11
        nodes = []
        for node in self.node:
            # fix ==> Context: Bad node spec for node. Name: /Constant OpType: Constant
            # or use onnxoptimizer
            # python -m onnxoptimizer /workspaces/python/onnx2/resnet18.onnx /workspaces/python/onnx2/resnet18_opt.onnx
            attributes = copy.deepcopy(node.attributes)
            if "Constant" == node.op_type:
                for key in attributes:
                    if type(node.attributes[key]) == np.ndarray:
                        # import pdb; pdb.set_trace()
                        if hasattr(onnx, "numpy_helper"):
                            attributes[key] = onnx.numpy_helper.from_array(
                                node.attributes[key]
                            )
                        else:
                            attributes[key] = self.from_array(node.attributes[key])
                    else:
                        attributes[key] = node.attributes[key]
            # import pdb; pdb.set_trace()
            node = helper.make_node(
                node.op_type,  # op_type
                node.input,  # inputs
                node.output,  # outputs
                node.name,  # name
                **attributes,  # attributes
            )
            nodes.append(node)
        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            nodes,  # nodes
            "test-model",  # name
            inputs,  # inputs
            outputs,  # outputs
            initializers,  # initializers
            value_info=value_infos,  # value_infos
        )
        # Create the model (ModelProto)
        model_def = helper.make_model(graph_def, producer_name="onnx-example")
        # print('The model is:\n{}'.format(model_def))
        # onnx.checker.check_model(model_def)
        # print('The model is checked!')
        with open(onnx_path, "wb") as f:
            f.write(model_def.SerializeToString())

    def from_array(self, arr: np.ndarray, name: str = None) -> TensorProto:
        assert arr.dtype != object
        tensor = TensorProto()
        if name:
            tensor.name = name
        tensor.dims.extend(arr.shape)
        tensor.data_type = self.ntype2otype[arr.dtype]
        tensor.raw_data = arr.tobytes()
        return tensor

    def __str__(self):
        s = f"name:{self.name}\n"
        s += f"input:\n"
        for tensor in self.input:
            s += str(tensor) + "\n"
        s += f"outut:\n"
        for tensor in self.output:
            s += str(tensor) + "\n"
        s += f"value_info:\n"
        for tensor in self.value_info:
            s += str(tensor) + "\n"
        s += f"node:\n"
        for node in self.node:
            s += str(node) + "\n"
        s += f"initializer:\n"
        for tensor in self.initializer:
            s += str(tensor) + "\n"
        return s

    def get_nodes(self):
        return self.node

    def get_input(self):  # (name, dtype, shape)
        return self.input

    def get_output(self):  # (name, dtype, shape)
        return self.output

    def get_initializer(self):  # (init.name, init_tensor)
        return self.initializer

    def get_node_byname(self, name: str) -> Node:
        for node in self.node:
            if node.name == name:
                return node
        return None

    def get_node_byoutput(self, name: str) -> List[Node]:
        find_nodes = []
        for node in self.node:
            for outp in node.output:
                if outp == name:
                    find_nodes.append(node)
        return find_nodes

    def get_node_byinput(self, name: str) -> List[Node]:
        find_nodes = []
        for node in self.node:
            for inp in node.input:
                if inp == name:
                    find_nodes.append(node)
                    break
        return find_nodes

    def get_init_byname(self, name: str) -> Tensor:
        for init in self.initializer:
            if init.name == name:
                return init
        return None

    def get_input_byname(self, name: str) -> Tensor:
        for inp in self.input:
            if inp.name == name:
                return inp
        return None

    def get_output_byname(self, name: str) -> Tensor:
        for outp in self.output:
            if outp.name == name:
                return outp
        return None

    def get_value_info_byname(self, name: str) -> Tensor:
        for vi in self.value_info:
            if vi.name == name:
                return vi
        return None

    def remove_node_byname(self, name: str) -> Node:
        for node in self.node:
            if node.name == name:
                return node
            self.node.remove(node)

    def remove_init_byname(self, name: str) -> Tensor:
        for init in self.initializer:
            if init.name == name:
                self.initializer.remove(init)

    def remove_input_byname(self, name: str) -> Tensor:
        for inp in self.input:
            if inp.name == name:
                self.input.remove(inp)

    def remove_output_byname(self, name: str) -> Tensor:
        for outp in self.output:
            if outp.name == name:
                self.output.remove(outp)


if __name__ == "__main__":
    graph = Graph(
        "/data/user/yswang/task/trunet/libonnx/examples/tinyexec_test/model/resnet18_opt.onnx"
    )
    print(graph)
    graph.save_onnx("model.onnx")

__all__ = ['Graph', 'Node', 'Tensor']