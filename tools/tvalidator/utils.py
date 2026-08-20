import onnx
import onnx.mapping
import numpy as np
import torch
from typing import Dict, List, Tuple
from collections import deque
from onnx import numpy_helper
import math
from dataclasses import dataclass
from contextlib import contextmanager
import os
import json

class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'

TRANSPARENT_OPS = {'Reshape', 'Transpose', 'Gather', 'Squeeze', 'Unsqueeze', 'Slice', 'Split', 'Concat', 'MaxPool',\
                    'Relu', 'Clip', 'Prelu', 'Resize'}

def parse_attribute_and_name(node):
        node_attribute = dict()
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.AttributeType.INTS:
                node_attribute[attr.name] = tuple(attr.ints)
            elif attr.type == onnx.AttributeProto.AttributeType.INT:
                node_attribute[attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.AttributeType.FLOAT:
                node_attribute[attr.name] = attr.f
            elif attr.type == onnx.AttributeProto.AttributeType.FLOATS:
                node_attribute[attr.name] = tuple(attr.floats)
            elif attr.type == onnx.AttributeProto.AttributeType.STRING:
                node_attribute[attr.name] = attr.s.decode('utf-8')
            elif attr.type == onnx.AttributeProto.AttributeType.TENSOR:
                value = numpy_helper.to_array(attr.t)
                node_attribute[attr.name] = value.item() if value.ndim == 0 else value.tolist()
            elif attr.type == onnx.AttributeProto.AttributeType.GRAPH:
                node_attribute[attr.name] = attr.g
            else:
                raise KeyError(
                            "The current operator({}) attribute({}) type is not supported,only support [float,int,ints,string,tensor,graph]".format(node.name,attr.name)
                        )
        return node.name, node_attribute

@contextmanager
def pushd(path):
    old = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)

@dataclass
class InputInfo:
    dtype: type
    shape: list


class ONNXModel:
    def __init__(self, onnx_path, dynamic_cfg=None, specified_inputs=None):
        self.model = onnx.load(onnx_path)
        self.dynamic_cfg = dynamic_cfg

        self.graph_input = None
        self._get_graph_input()

        self.input_info = {}
        self.inputs = specified_inputs
        self.dynamic_values = {}

        self._init_quant_op_configs()

    def _get_graph_input(self):
        init_names = {init.name for init in self.model.graph.initializer}
        self.graph_input = [inp for inp in self.model.graph.input if inp.name not in init_names]

    def _get_input_info(self):
        graph_inputs = self.graph_input
        symbol_dict = {}
        rng = np.random.default_rng()

        def resolve_dimension(symbol):
            if not symbol:
                raise ValueError("Dynamic input dimension is unnamed")
            if symbol in symbol_dict:
                return symbol_dict[symbol]
            if self.dynamic_cfg is None:
                raise ValueError(f"Missing --cfg range for dynamic dimension '{symbol}'")
            if symbol in self.dynamic_cfg:
                min_val, max_val, factor = self.dynamic_cfg[symbol]
                first_valid = ((min_val + factor - 1) // factor) * factor
                candidate_count = (max_val - first_valid) // factor + 1
                value = first_valid + int(rng.integers(candidate_count)) * factor
                symbol_dict[symbol] = value
                print(
                    f"  Symbol <{symbol}> is set to <{value}> "
                    f"(range={min_val}:{max_val}, factor={factor})"
                )
                return value

            for base_symbol in self.dynamic_cfg:
                resolve_dimension(base_symbol)
            try:
                value = int(eval(symbol, {
                    "__builtins__": None,
                    "floor": math.floor,
                    "ceil": math.ceil,
                    "Max": max,
                    "Min": min,
                    **symbol_dict,
                }))
            except Exception as exc:
                raise ValueError(
                    f"Missing --cfg range for dynamic dimension '{symbol}'"
                ) from exc
            if value <= 0:
                raise ValueError(f"Dynamic dimension '{symbol}' resolved to {value}")
            symbol_dict[symbol] = value
            print(f"  Symbol <{symbol}> is resolved to <{value}>")
            return value

        for i, vi in enumerate(graph_inputs):
            name = vi.name
            tensor_type = vi.type.tensor_type
            dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[tensor_type.elem_type]
            if self.inputs is not None:
                shape = list(self.inputs[i].shape)
                expected_rank = len(tensor_type.shape.dim)
                if len(shape) != expected_rank:
                    raise ValueError(
                        f"Input '{name}' rank {len(shape)} does not match model rank {expected_rank}"
                    )
                for dim_index, (actual, dim) in enumerate(zip(shape, tensor_type.shape.dim)):
                    if dim.dim_value > 0:
                        if actual != dim.dim_value:
                            raise ValueError(
                                f"Input '{name}' dimension {dim_index} is {actual}, expected {dim.dim_value}"
                            )
                        continue
                    symbol = dim.dim_param
                    if symbol in (self.dynamic_cfg or {}):
                        min_val, max_val, factor = self.dynamic_cfg[symbol]
                        if actual < min_val or actual > max_val or actual % factor != 0:
                            raise ValueError(
                                f"Input '{name}' dimension '{symbol}'={actual} violates "
                                f"--cfg {min_val}:{max_val}:{factor}"
                            )
                        if symbol in symbol_dict and symbol_dict[symbol] != actual:
                            raise ValueError(
                                f"Dynamic dimension '{symbol}' has inconsistent values "
                                f"{symbol_dict[symbol]} and {actual}"
                            )
                        symbol_dict[symbol] = actual
                        continue
                    expected = resolve_dimension(symbol)
                    if actual != expected:
                        raise ValueError(
                            f"Input '{name}' dimension '{symbol}' is {actual}, expected {expected}"
                        )
            else:
                shape = []
                for d in tensor_type.shape.dim:
                    if d.dim_value > 0:
                        shape.append(d.dim_value)
                    else:
                        assert self.dynamic_cfg is not None, "Dynamic symbol info must be provided!"
                        symbol = d.dim_param
                        shape.append(resolve_dimension(symbol))
            self.input_info[name] = InputInfo(dtype, shape)
        self.dynamic_values = symbol_dict

    def specialize_dynamic_reshapes(self):
        if not self.dynamic_values:
            return

        eval_globals = {
            "__builtins__": None,
            "floor": math.floor,
            "ceil": math.ceil,
            "Max": max,
            "Min": min,
            **self.dynamic_values,
        }
        metadata = {prop.key: prop.value for prop in self.model.metadata_props}
        dynamic_initializers = json.loads(
            metadata.get("thinker_dynamic_initializers", "{}")
        )
        initializer_indices = {
            initializer.name: index
            for index, initializer in enumerate(self.model.graph.initializer)
        }
        for name, spec in dynamic_initializers.items():
            index = initializer_indices.get(name)
            if index is None:
                continue
            try:
                values = np.asarray(
                    [int(eval(expr, eval_globals)) for expr in spec["expressions"]],
                    dtype=np.int64,
                ).reshape(spec["shape"])
            except Exception as exc:
                raise ValueError(
                    f"Failed to specialize dynamic initializer '{name}': {exc}"
                ) from exc
            replacement = numpy_helper.from_array(values, name)
            self.model.graph.initializer[index].CopyFrom(replacement)
            print(f"  Updated dynamic initializer <{name}> to {values.tolist()}")

        value_info = {
            value.name: value
            for value in list(self.model.graph.value_info) + list(self.model.graph.output)
        }
        for node in self.model.graph.node:
            if node.op_type != "Reshape" or len(node.input) < 2 or not node.output:
                continue
            shape_index = initializer_indices.get(node.input[1])
            output_info = value_info.get(node.output[0])
            if shape_index is None or output_info is None:
                continue

            shape = numpy_helper.to_array(self.model.graph.initializer[shape_index]).copy()
            dims = output_info.type.tensor_type.shape.dim
            if shape.ndim != 1 or len(shape) != len(dims):
                continue

            changed = False
            for index, dim in enumerate(dims):
                if shape[index] <= 0:
                    continue
                if dim.dim_value > 0:
                    value = dim.dim_value
                elif dim.dim_param:
                    try:
                        value = int(eval(dim.dim_param, eval_globals))
                    except Exception as exc:
                        raise ValueError(
                            f"Failed to evaluate dynamic shape '{dim.dim_param}' "
                            f"for Reshape node '{node.name}': {exc}"
                        ) from exc
                else:
                    continue
                if value <= 0:
                    raise ValueError(
                        f"Dynamic shape for Reshape node '{node.name}' resolved to {value}"
                    )
                if shape[index] != value:
                    shape[index] = value
                    changed = True

            if changed:
                replacement = numpy_helper.from_array(
                    shape,
                    self.model.graph.initializer[shape_index].name,
                )
                self.model.graph.initializer[shape_index].CopyFrom(replacement)
                print(f"  Updated dynamic Reshape <{node.name}> shape to {shape.tolist()}")

    def get_platform(self):
        for node in self.model.graph.node:
            _, attrs = parse_attribute_and_name(node)
            if "platform" in attrs:
                return attrs['platform']
        return "venus"

    def generate_input(self):
        def resolve_input_index(node, locator_logic: dict):
            logic_type = locator_logic.get('type')
            if logic_type == 'static': return locator_logic.get('index')
            if logic_type == 'conditional':
                arg, node_arg_val = locator_logic.get('arg'), len(node.input)
                if arg == 'num_inputs':
                    for case in locator_logic.get('cases', []):
                        if 'if_equal' in case and node_arg_val == case['if_equal']: return case['index']
                        if 'if_greater_equal' in case and node_arg_val >= case['if_greater_equal']: return case['index']
            return None

        self._get_input_info()

        model = self.model
        graph = model.graph

        consumer_map: Dict[str, List[Tuple[onnx.NodeProto, int]]] = {i.name: [] for i in self.graph_input}
        inits_map = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
        tensor_info = {
            value.name: value.type.tensor_type
            for value in list(graph.input) + list(graph.value_info) + list(graph.output)
        }
        for initializer in graph.initializer: consumer_map[initializer.name] = []
        for node in graph.node:
            for i, inp in enumerate(node.input):
                if inp not in consumer_map: consumer_map[inp] = []
                consumer_map[inp].append((node, i))

        inputs_dict = {}
        processed_graph_inputs = set()

        def random_input(info, low=-128, high=128):
            if np.issubdtype(info.dtype, np.integer):
                return np.random.randint(low, high, size=info.shape, dtype=info.dtype)
            if np.issubdtype(info.dtype, np.floating):
                return np.random.uniform(low, high, size=info.shape).astype(info.dtype)
            raise TypeError(f"Unsupported input dtype: {info.dtype}")
        
        print(f"  Starting forward search from graph inputs...{Colors.RESET}")
        for i, graph_input in enumerate(self.graph_input):
            if graph_input.name in processed_graph_inputs: continue

            print(f"\n  Processing path starting from input: '{graph_input.name}'")

            queue = deque([(graph_input.name, graph_input.name)])
            visited_tensors = {graph_input.name}
            input_ready = False
            while queue and not input_ready: 
                current_tensor, original_source = queue.popleft()
                consumers = consumer_map.get(current_tensor, [])
                
                for consumer_node, consumer_index in consumers:
                    if consumer_node.op_type in self.__quant_op_configs:
                        print(f"    -> Path reached potential target '{consumer_node.name}' at its input index {consumer_index}.")
                        config = self.__quant_op_configs[consumer_node.op_type]
                        
                        # Dynamically check if the connection is to a quantizable slot
                        matched_quant_input_info = None
                        for quant_input_info in config['quantizable_inputs']:
                            actual_index = resolve_input_index(consumer_node, quant_input_info['locator_logic'])
                            if actual_index == consumer_index:
                                matched_quant_input_info = quant_input_info
                                break
                        
                        if matched_quant_input_info:
                            print(f"    -> SUCCESS: Connection matches the defined quantizable input '{matched_quant_input_info['name']}'.")
                            _, attrs = parse_attribute_and_name(consumer_node)
                            scale_val = attrs.get(matched_quant_input_info['scale_attr'], None)
                            zp_val = attrs.get(matched_quant_input_info['zp_attr'], (0.0))
                            data_bits = attrs.get('data_bits', 8)
                            if scale_val is None or zp_val is None:
                                print(f"    -> ERROR: Could not extract quant params. Skipping.")
                                continue

                            print(f"    -> ACTION: generate quantizable input <{original_source}>, shape is {self.input_info[original_source].shape}.")
                            if self.inputs is not None:
                                thinker_input = self.inputs[i]
                            else:
                                if data_bits == 8:
                                    data_dtype = np.int8
                                elif data_bits == 16:
                                    data_dtype = np.int16
                                else:
                                    data_dtype = np.int32
                                
                                if consumer_node.op_type == "LSTMInt" and matched_quant_input_info['scale_attr'] == "scale_c": ###LSTMInt 算子的cell固定为int32类型，但实际数据为int16
                                    data_bits = 16
                                    data_dtype = np.int32
                            
                                bound_val = math.pow(2, data_bits-1)
                                thinker_input = np.random.randint(-bound_val, bound_val, size = self.input_info[original_source].shape, dtype=data_dtype)
                            onnxrunner_input = torch.from_numpy((thinker_input - zp_val).astype(np.float32) / scale_val).cpu()
                            inputs_dict[original_source] = (onnxrunner_input, thinker_input)
                            print(f"    -> SUCCESS: quantizable input {original_source} is generated.")

                            processed_graph_inputs.add(original_source)
                            input_ready = True
                            break # Break from consumers loop, this path is done
                        else:
                            # input do not need to be quantized
                            if self.inputs is not None:
                                print(f"    -> ACTION: use provided input <{original_source}>, shape is {self.input_info[original_source].shape}.")
                                thinker_input = self.inputs[i]
                            else:
                                print(f"    -> ACTION: generate normal input <{original_source}>, shape is {self.input_info[original_source].shape}.")
                                if consumer_node.op_type == "LSTMInt" and consumer_index == 1:
                                    _, attrs = parse_attribute_and_name(consumer_node)
                                    sequence_axis = 1 if attrs.get("batch_first", 0) else 0
                                    sequence_name = consumer_node.input[0]
                                    if sequence_name in self.input_info:
                                        sequence_length = self.input_info[sequence_name].shape[sequence_axis]
                                    else:
                                        sequence_dim = tensor_info[sequence_name].shape.dim[sequence_axis]
                                        sequence_length = sequence_dim.dim_value
                                        if sequence_length <= 0:
                                            sequence_length = self.dynamic_values[sequence_dim.dim_param]
                                    thinker_input = np.full(
                                        self.input_info[original_source].shape,
                                        sequence_length,
                                        dtype=self.input_info[original_source].dtype,
                                    )
                                else:
                                    thinker_input = random_input(self.input_info[original_source])
                            onnxrunner_input = torch.from_numpy(thinker_input).cpu()
                            inputs_dict[original_source] = (onnxrunner_input, thinker_input)
                            print(f"    -> SUCCESS: normal input <{original_source}> generated.")

                            processed_graph_inputs.add(original_source)
                            input_ready = True
                            break # Break from consumers loop, this path is done
                    elif consumer_node.op_type in TRANSPARENT_OPS:
                        if consumer_node.op_type == "Gather" and consumer_index==1: ##输入是indices，原生就为定点值，非量化数据
                            # input do not need to be quantized
                            if self.inputs is not None:
                                print(f"    -> ACTION: use provided input <{original_source}>, shape is {self.input_info[original_source].shape}.")
                                thinker_input = self.inputs[i]
                            else:
                                print(f"    -> ACTION: generate normal input <{original_source}>, shape is {self.input_info[original_source].shape}.")
                                min_val = 0
                                max_val = inits_map[consumer_node.input[0]].shape[0]
                                thinker_input = np.random.randint(min_val, max_val, size=self.input_info[original_source].shape, dtype=self.input_info[original_source].dtype)
                            onnxrunner_input = torch.from_numpy(thinker_input).cpu()
                            inputs_dict[original_source] = (onnxrunner_input, thinker_input)
                            print(f"    -> SUCCESS: normal input <{original_source}> generated.")

                            processed_graph_inputs.add(original_source)
                            input_ready = True
                            break
                        for output_tensor in consumer_node.output:
                            if output_tensor not in visited_tensors:
                                print(f"    -> Traversing through transparent op '{consumer_node.name}'...")
                                visited_tensors.add(output_tensor)
                                queue.append((output_tensor, original_source))
                if input_ready:
                    break
            if not input_ready:
                print(f"    -> ACTION: use normal input <{original_source}>, shape is {self.input_info[original_source].shape}.")
                if self.inputs is not None:
                    thinker_input = self.inputs[i]
                else:
                    thinker_input = random_input(self.input_info[original_source])
                onnxrunner_input = torch.from_numpy(thinker_input).cpu()
                inputs_dict[original_source] = (onnxrunner_input, thinker_input)
                processed_graph_inputs.add(original_source)
                print(f"    -> SUCCESS: normal input <{original_source}> generated.")
        return inputs_dict
        
    def _init_quant_op_configs(self):
        self.__quant_op_configs = {
            'AvgPool2dInt': {
                'quantizable_inputs': [
                    {
                        'name': 'input',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'data_zero_point'
                    }
                ]
            },
            'BmmInt': {
                'quantizable_inputs': [
                    {
                        'name': 'input_x',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'input_x_zero_point'
                    },
                    {
                        'name': 'input_y',
                        'locator_logic': {'type': 'static', 'index': 1},
                        'scale_attr': 'scale_y',
                        'zp_attr': 'input_y_zero_point'
                    }
                ]
            },
            'Conv1dInt': {
                'quantizable_inputs': [
                    {
                        'name': 'input',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'data_zero_point'
                    }
                ]
            },
            'Conv2dInt': {
                'quantizable_inputs': [
                    {
                        'name': 'input',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'data_zero_point'
                    }
                ]
            },
            'ConvTranspose2dInt': {
                'quantizable_inputs': [
                    {
                        'name': 'input',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'data_zero_point'
                    }
                ]
            },
            'GRUInt': {
                'quantizable_inputs': [
                    {'name': 'sequence_input', 
                     'locator_logic': {'type': 'static', 'index': 0}, 
                     'scale_attr': 'scale_x', 
                     'zp_attr': 'x_zero_point'},
                    {'name': 'initial_hidden', 
                     'locator_logic': {'type': 'conditional', 'arg': 'num_inputs', 'cases': [{'if_equal': 6, 'index': 1}, {'if_equal': 7, 'index': 1}, {'if_equal': 8, 'index': 2}]},
                     'scale_attr': 'scale_h', 
                     'zp_attr': 'h_zero_point'},
                ]
            },
            'QGRU': {
                'quantizable_inputs': [
                    {'name': 'sequence_input', 
                     'locator_logic': {'type': 'static', 'index': 0}, 
                     'scale_attr': 'scale_x', 
                     'zp_attr': 'x_zero_point'},
                    {'name': 'initial_hidden', 
                     'locator_logic': {'type': 'conditional', 'arg': 'num_inputs', 'cases': [{'if_equal': 6, 'index': 1}, {'if_equal': 7, 'index': 1}, {'if_equal': 8, 'index': 2}]},
                     'scale_attr': 'scale_h', 
                     'zp_attr': 'h_zero_point'},
                ]
            },
            'iqAdd': {
                'quantizable_inputs': [
                    {
                        'name': 'input_x',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'input_x_zero_point'
                    },
                    {
                        'name': 'input_y',
                        'locator_logic': {'type': 'static', 'index': 1},
                        'scale_attr': 'scale_y',
                        'zp_attr': 'input_y_zero_point'
                    }
                ]
            },
            'iqCat': {
                'quantizable_inputs': [
                    {
                        'name': 'input_0',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x_0',
                        'zp_attr': 'input_zero_point_0'
                    },
                    {
                        'name': 'input_1',
                        'locator_logic': {'type': 'static', 'index': 1},
                        'scale_attr': 'scale_x_1',
                        'zp_attr': 'input_zero_point_1'
                    },
                    {
                        'name': 'input_2',
                        'locator_logic': {'type': 'static', 'index': 2},
                        'scale_attr': 'scale_x_2',
                        'zp_attr': 'input_zero_point_2'
                    },
                    {
                        'name': 'input_3',
                        'locator_logic': {'type': 'static', 'index': 3},
                        'scale_attr': 'scale_x_3',
                        'zp_attr': 'input_zero_point_3'
                    }
                ]
            },
            'iqDiv': {
                'quantizable_inputs': [
                    {
                        'name': 'input_x',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'input_x_zero_point'
                    },
                    {
                        'name': 'input_y',
                        'locator_logic': {'type': 'static', 'index': 1},
                        'scale_attr': 'scale_y',
                        'zp_attr': 'input_y_zero_point'
                    }
                ]
            },
            'iqMul': {
                'quantizable_inputs': [
                    {
                        'name': 'input_x',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'input_x_zero_point'
                    },
                    {
                        'name': 'input_y',
                        'locator_logic': {'type': 'static', 'index': 1},
                        'scale_attr': 'scale_y',
                        'zp_attr': 'input_y_zero_point'
                    }
                ]
            },
            'iqSigmoid': {
                'quantizable_inputs': [
                    {
                        'name': 'input',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'data_zero_point'
                    }
                ]
            },
            'iqSum': {
                'quantizable_inputs': [
                    {
                        'name': 'input',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'data_zero_point'
                    }
                ]
            },
            'LayerNormInt': {
                'quantizable_inputs': [
                    {
                        'name': 'input',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'data_zero_point'
                    }
                ]
            },
            'LinearInt': {
                'quantizable_inputs': [
                    {
                        'name': 'input',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'data_zero_point'
                    }
                ]
            },
            'LSTMInt': {
                'quantizable_inputs': [
                    {'name': 'sequence_input', 
                     'locator_logic': {'type': 'static', 'index': 0}, 
                     'scale_attr': 'scale_x', 
                     'zp_attr': 'i_zero_point'},
                    {'name': 'initial_hidden', 
                     'locator_logic': {'type': 'conditional', 'arg': 'num_inputs', 'cases': [{'if_equal': 7, 'index': 1}, {'if_equal': 8, 'index': 2}]}, 
                     'scale_attr': 'scale_h', 
                     'zp_attr': 'h_zero_point'},
                    {'name': 'initial_cell', 
                     'locator_logic': {'type': 'conditional', 'arg': 'num_inputs', 'cases': [{'if_equal': 7, 'index': 2}, {'if_equal': 8, 'index': 3}]}, 
                     'scale_attr': 'scale_c', 
                     'zp_attr': 'c_zero_point'}
                ]
            },
            'Quant': {
                'quantizable_inputs': [
                    {
                        'name': 'input',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'zero_point'
                    }
                ]
            },
            'SoftmaxInt': {
                'quantizable_inputs': [
                    {
                        'name': 'input',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'data_zero_point'
                    }
                ]
            },
            'QGelu': {
                'quantizable_inputs': [
                    {
                        'name': 'input',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'data_zero_point'
                    }
                ]
            },
        }

__all__ = ['Colors', 'ONNXModel']
