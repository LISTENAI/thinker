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

class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'

TRANSPARENT_OPS = {'Reshape', 'Transpose', 'Gather', 'Squeeze', 'Unsqueeze', 'Slice', 'Split', 'MaxPool',\
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
                node_attribute[attr.name] = list(numpy_helper.to_array(node.attribute[0].t))
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

        self._init_quant_op_configs()

    def _get_graph_input(self):
        init_names = {init.name for init in self.model.graph.initializer}
        self.graph_input = [inp for inp in self.model.graph.input if inp.name not in init_names]

    def _get_input_info(self):
        graph_inputs = self.graph_input
        symbol_dict = {}

        for i, vi in enumerate(graph_inputs):
            name = vi.name
            tensor_type = vi.type.tensor_type
            dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[tensor_type.elem_type]
            if self.inputs is not None:
                shape = list(self.inputs[i].shape)
            else:
                shape = []
                for d in tensor_type.shape.dim:
                    if d.dim_value > 0:
                        shape.append(d.dim_value)
                    else:
                        assert self.dynamic_cfg is not None, "Dynamic symbol info must be provided!"
                        symbol = d.dim_param
                        if symbol in symbol_dict:
                            shape.append(symbol_dict[symbol])
                        else:
                            cfg = self.dynamic_cfg[symbol]
                            if isinstance(cfg, tuple):
                                step = cfg[2]
                                min_val = cfg[0] / step
                                max_val = cfg[1]/step + 1
                            else:
                                step = 1
                                min_val = 1
                                max_val = cfg + 1
                            rng = np.random.default_rng()
                            val = rng.integers(min_val, max_val) * step
                            symbol_dict[symbol] = val
                            shape.append(val)
                            print(f"  Symbol <{symbol}> is set to <{val}>")
            self.input_info[name] = InputInfo(dtype, shape)

    def get_platform(self):
        for node in self.model.graph.node:
            _, attrs = parse_attribute_and_name(node)
            if "platform" in attrs:
                return attrs['platform']
            else:
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
        for initializer in graph.initializer: consumer_map[initializer.name] = []
        for node in graph.node:
            for i, inp in enumerate(node.input):
                if inp not in consumer_map: consumer_map[inp] = []
                consumer_map[inp].append((node, i))

        inputs_dict = {}
        processed_graph_inputs = set()
        
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
                                thinker_input = np.random.randint(-128, 128, size=self.input_info[original_source].shape, dtype=self.input_info[original_source].dtype)
                            onnxrunner_input = torch.from_numpy(thinker_input).cpu()
                            inputs_dict[original_source] = (onnxrunner_input, thinker_input)
                            print(f"    -> SUCCESS: normal input <{original_source}> generated.")

                            processed_graph_inputs.add(original_source)
                            input_ready = True
                            break # Break from consumers loop, this path is done
                    elif consumer_node.op_type in TRANSPARENT_OPS:
                        for output_tensor in consumer_node.output:
                            if output_tensor not in visited_tensors:
                                print(f"    -> Traversing through transparent op '{consumer_node.name}'...")
                                visited_tensors.add(output_tensor)
                                queue.append((output_tensor, original_source))
                    else:
                        raise ValueError(f"  Generate input <{graph_input.name}> failed.")
                if input_ready:
                    break
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
                     'locator_logic': {'type': 'conditional', 'arg': 'num_inputs', 'cases': [{'if_equal': 7, 'index': 1}, {'if_equal': 8, 'index': 2}]}, 
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
                     'scale_attr': 'scale_i', 
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
        }

__all__ = ['Colors', 'ONNXModel']