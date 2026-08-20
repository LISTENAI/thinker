import numpy as np
from pathlib import Path
import sys
import os
import subprocess
import onnx
from onnx import ModelProto
from typing import Dict, List
from .utils import ONNXModel, Colors
from .thinker_runner import ThinkerRunner
from linger.checker import OnnxRunner
import shutil
import argparse
import re
import torch

def _cmake_bool(value) -> str:
    if isinstance(value, str):
        return "ON" if value.lower() in ("1", "on", "true", "yes") else "OFF"
    return "ON" if value else "OFF"

def _remove_if_exists(path: Path):
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

def _move_dump_dir(src: Path, dst: Path):
    if not src.exists():
        raise FileNotFoundError(f"Expected dump directory does not exist: {src}")

    # _remove_if_exists(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))

def _safe_filename(name: str) -> str:
    """
    Convert ONNX input name to a safe filename.
    For example:
        "input.1"      -> "input.1"
        "images/input" -> "images_input"
    """
    return re.sub(r'[\\/:*?"<>|\s]+', "_", name)

class Validator:
    def __init__(self, model, linger_dir:str, thinker_dir:str, tensor_shapes:Dict):
        if isinstance(model, str):
            self.model = onnx.load(model)
        elif isinstance(model, ModelProto):
            self.model = model
        else:
            raise RuntimeError(
                f"Unsupported model type: {type(model)}. "
                f"Expected a ONNX path (str) or onnx.ModelProto."
            )
        self.linger_dir = Path(linger_dir)
        self.thinker_dir = Path(thinker_dir)
        self.tensor_shapes = tensor_shapes

    @staticmethod
    def create_prefix_map(directory: Path) -> Dict[str, Path]:
        if not directory.is_dir():
            print(f"Error: Directory not found at '{directory}'", file=sys.stderr)
            sys.exit(1)
            
        prefix_map = {}
        for file_path in directory.iterdir():
            if file_path.is_file() and '##' in file_path.name:
                prefix = file_path.name.split('##', 1)[0]
                if prefix in prefix_map:
                    print(f"Warning: Duplicate prefix '{prefix}' found in directory '{directory}'. Skipping file '{file_path.name}'.", file=sys.stderr)
                else:
                    prefix_map[prefix] = file_path
        return prefix_map
    
    def try_open_vscode_diff(self, file1, file2):
        if shutil.which("code") is None:
            return 
        
        print(f"  -> Launching VSCode compare for the following files: ")
        print(f"      -> Linger : {file1}")
        print(f"      -> Thinker: {file2}")

        try:
            subprocess.run(
                ["code", "--diff", str(file1), str(file2)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception:
            pass
    
    @staticmethod
    def compare_file_contents(file1: Path, file2: Path) -> bool:
        try:
            with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
                # Read all lines, strip whitespace, and filter out empty lines.
                # Convert to a set to make the comparison order-agnostic.
                content1 = [line.strip() for line in f1 if line.strip()]
                content2 = [line.strip() for line in f2 if line.strip()]
                return content1 == content2
        except IOError as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            return False

    def compare(self):
        """Compare files between two dump directories."""
        print(f"  -> Starting comparison between '{self.linger_dir}' and '{self.thinker_dir}'")

        dequant_scales = {}
        argmax_outputs = set()
        high_precision_scales = {}
        high_precision_tensors = {}
        for node in self.model.graph.node:
            attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
            if node.op_type == "Dequant":
                if node.output and "scale_o" in attrs:
                    dequant_scales[node.output[0].replace("/", "_")] = float(attrs["scale_o"])
            elif node.op_type == "ArgMax" and node.output:
                argmax_outputs.add(node.output[0].replace("/", "_"))
            elif node.output and attrs.get("o_bits", 0) >= 16 and "scale_o" in attrs:
                scale_o = attrs["scale_o"]
                if isinstance(scale_o, (list, tuple)):
                    scale_o = scale_o[0]
                scale_o = float(scale_o)
                if scale_o >= 2**20:
                    scale_info = (
                        scale_o / 2**20,
                        int(attrs["o_bits"]),
                    )
                    high_precision_tensors[node.output[0]] = scale_info
                    high_precision_scales[node.output[0].replace("/", "_")] = scale_info
            elif node.op_type == "Transpose" and node.input and node.input[0] in high_precision_tensors:
                scale_info = high_precision_tensors[node.input[0]]
                for output in node.output:
                    high_precision_tensors[output] = scale_info
                    high_precision_scales[output.replace("/", "_")] = scale_info

        # Build prefix maps
        dir1_map = self.create_prefix_map(self.linger_dir)
        dir2_map = self.create_prefix_map(self.thinker_dir)

        dir1_prefixes = set(dir1_map.keys())
        dir2_prefixes = set(dir2_map.keys())
        common_prefixes = dir1_prefixes.intersection(dir2_prefixes)

        identical_files: List[tuple] = []
        different_files: List[tuple] = []

        if not common_prefixes:
            print("\n ❗ No corresponding files found to compare.")
            return

        # Compare corresponding files
        for prefix in sorted(list(common_prefixes)):
            file1 = dir1_map[prefix]
            file2 = dir2_map[prefix]

            if prefix in dequant_scales:
                linger_data = np.loadtxt(file1, dtype=np.float64)
                thinker_data = np.loadtxt(file2, dtype=np.float64)
                identical = np.array_equal(linger_data, np.rint(thinker_data * dequant_scales[prefix]))
            elif prefix in argmax_outputs:
                linger_data = np.atleast_1d(np.loadtxt(file1, dtype=np.int64))
                thinker_data = np.atleast_1d(np.loadtxt(file2, dtype=np.int64))
                if thinker_data.size == linger_data.size * 2:
                    thinker_data = thinker_data[linger_data.size:]
                identical = np.array_equal(linger_data, thinker_data)
            elif prefix in high_precision_scales:
                scale, bits = high_precision_scales[prefix]
                linger_data = np.atleast_1d(np.loadtxt(file1, dtype=np.int64))
                thinker_data = np.atleast_1d(np.loadtxt(file2, dtype=np.int64))
                linger_data = np.rint(linger_data * scale)
                linger_data = np.clip(linger_data, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
                # Fixed-point implementations can differ by one final LSB
                # after the last requantization/rounding step.
                identical = np.all(np.abs(linger_data - thinker_data) <= 1)
            else:
                identical = self.compare_file_contents(file1, file2)

            if identical:
                identical_files.append((file1.name, file2.name))
            else:
                different_files.append((file1.name, file2.name))

        tensor_topo_order = {}

        for idx, node in enumerate(self.model.graph.node):
            for out in node.output:
                tensor_name = out.replace("/", "_")
                tensor_topo_order[tensor_name] = idx

        def get_prefix_from_filename(filename: str) -> str:
            return filename.split("##", 1)[0]

        def topo_key(pair):
            f1, f2 = pair
            prefix = get_prefix_from_filename(f1)
            return tensor_topo_order.get(prefix, 10**9)

        different_files.sort(key=topo_key)

        # Print summary
        if different_files:
            print(f"❌ {Colors.RED}Consistency verification failed!{Colors.RESET}")

            first_f1_name, first_f2_name = different_files[0]

            first_f1 = self.linger_dir / first_f1_name
            first_f2 = self.thinker_dir / first_f2_name

            prefix = get_prefix_from_filename(first_f1_name)

            true_shape = self.tensor_shapes.get(prefix, None)

            print(f"  -> [!] First mismatch tensor: {prefix}")

            if true_shape is None:
                print(f"   No shape metadata found for tensor '{prefix}', cannot reshape.")
            else:
                print(f"  -> Shape: {true_shape}")

            def load_tensor_txt(fp: Path):
                with open(fp, "r", encoding="utf-8") as f:
                    data = [int(x.strip()) for x in f if x.strip() != ""]
                return data

            data1 = load_tensor_txt(first_f1)
            data2 = load_tensor_txt(first_f2)

            total_elems = len(data1)

            if true_shape is not None:
                try:
                    arr1 = np.array(data1).reshape(true_shape)
                    arr2 = np.array(data2).reshape(true_shape)
                except Exception:
                    arr1 = np.array(data1)
                    arr2 = np.array(data2)
                    print("   Reshape failed, using flat comparison instead.")
            else:
                arr1 = np.array(data1)
                arr2 = np.array(data2)

            if true_shape is not None and arr1.shape == tuple(true_shape):
                for idx in np.ndindex(*true_shape):  
                    if arr1[idx] != arr2[idx]:
                        mismatch_idx = idx 
                        break
            else:
                total_elems = arr1.size
                for i in range(total_elems):
                    if arr1.flat[i] != arr2.flat[i]:
                        mismatch_idx = i
                        break

            mismatch_indices = []

            if true_shape is not None and arr1.shape == tuple(true_shape):
                # 多维比较
                for idx in np.ndindex(*true_shape):
                    if arr1[idx] != arr2[idx]:
                        mismatch_indices.append(idx)
                        if len(mismatch_indices) >= 16:  # 只取前 16 个
                            break
            else:
                # 退化为一维
                total_elems = arr1.size
                for i in range(total_elems):
                    if arr1.flat[i] != arr2.flat[i]:
                        mismatch_indices.append(i)
                        if len(mismatch_indices) >= 16:
                            break

            if not mismatch_indices:
                print("   No differing indices found (unexpected).")
            else:
                print(f"\n    -> Showing first {len(mismatch_indices)} mismatched entries:")
                print("    " + "-"*65)
                print("    |     Index      |   Linger (training)   |  Thinker (inference) |")
                print("    " + "-"*65)

                for idx in mismatch_indices:
                    if isinstance(idx, tuple):  # 多维
                        v1 = arr1[idx]
                        v2 = arr2[idx]
                        idx_str = str(idx)
                    else:                       # 一维
                        v1 = arr1.flat[idx]
                        v2 = arr2.flat[idx]
                        idx_str = str(idx)

                    print(f"    | {idx_str:<14} | {str(v1):>10}            | {str(v2):>10}           |")

                print("    " + "-"*65)
            self.try_open_vscode_diff(first_f1, first_f2)
            # subprocess.run(["code", "--diff", first_f1, first_f2])
        else:
            print(f"✅ {Colors.GREEN}Consistency verification passed!{Colors.RESET}")

class ThinkerValidator:
    def __init__(self,onnx_path: str, model_resource_path:str=None, lib_path: str=None, inputs: List[np.ndarray]=None, 
                 dynamic_cfg: Dict = None, param_check: bool = True, runtime_check: bool = True):
        """
        Initializes the ThinkerValidator.

        Args:
            lib_path (str): Path to the Thinker library.
            onnx_path (str): Path to the ONNX model.
            model_resource_path (str, optional): Path to the model resource file.
                If left empty, the system will automatically perform model packaging.
            inputs (List[np.ndarray], optional): Input list provided by user, if left empty, the module will generate 
                random input data.
        """
        self.lib_path = lib_path
        self.onnx_path = onnx_path
        self.reference_onnx_path = onnx_path
        self.onnx_name = Path(onnx_path).stem

        self.workspace_dir = Path.cwd() / "workspace" / self.onnx_name
        _remove_if_exists(self.workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self.model_resource_path = model_resource_path
        self.platform = None
        self.dynamic_shape = False
        self.param_check = param_check
        self.runtime_check = runtime_check
        if dynamic_cfg is not None:
            self.dynamic_shape = True
            self.dynamic_cfg = dynamic_cfg

        self.tensor_shapes = None

        self.onnx_model = ONNXModel(onnx_path, dynamic_cfg, inputs)

        self._init_platform()

    def _init_platform(self):
        self.platform = self.onnx_model.get_platform()

    def _generate_input(self):
        input_data_dict = self.onnx_model.generate_input()
        linger_input, thinker_input = [], []

        save_dir = Path.cwd() / "workspace" / self.onnx_name

        for input in self.onnx_model.graph_input:
            li, ti = input_data_dict[input.name]
            linger_input.append(li)
            thinker_input.append(ti)

            safe_name = _safe_filename(input.name)

            linger_input_int_path = save_dir / f"{safe_name}_linger.npy"
            thinker_input_path = save_dir / f"{safe_name}_thinker.bin"

            # if isinstance(li, torch.Tensor):
            #     li_np = li.detach().cpu().numpy()
            # else:
            #     li_np = np.asarray(li)
                
            if isinstance(ti, torch.Tensor):
                ti_np = ti.detach().cpu().numpy()
            else:
                ti_np = np.asarray(ti)

            np.save(linger_input_int_path, ti_np)
            ti_np.tofile(thinker_input_path)

        return linger_input, thinker_input

    def run_linger_inference(self, input_data):
        (Path.cwd() / "data" / "onnxrunner_int").mkdir(parents=True, exist_ok=True)
        onnx_runner = OnnxRunner(self.reference_onnx_path, True)
        self.tensor_shapes = onnx_runner.get_tensor_info()
        res = onnx_runner.run(input_data)

    def _build_thinker(self):
        from .utils import pushd
        subprocess.run("rm -rf build", shell=True)

        with pushd("build"):
            cmake_cmd = [
                "cmake",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DTHINKER_SHARED_LIB=ON",
                "-DTHINKER_PROFILE=OFF",
                "-DTHINKER_RESULT_DUMP=ON",
                "-DTHINKER_RESULT_CRC_PRINT=OFF",
                "-DTHINKER_RESOUCR_CRC_CHECK=OFF",
                "-DTHINKER_USE_MTQ=OFF",
                "-DTHINKER_USE_NNBLAS=OFF",
                "-DTHINKER_TARGET_CHECK=OFF",
                f"-DTHINKER_PARAM_CHECK={_cmake_bool(self.param_check)}",
                f"-DTHINKER_RUNTIME_CHECK={_cmake_bool(self.runtime_check)}"
            ]

            platform_map = {
                "arcs": "ARCS",
                "venus": "VENUS",
                "venusa": "VENUSA",
            }
            target_platform = platform_map.get(self.platform.lower())
            if target_platform is None:
                raise RuntimeError(f"Unsupported platform: <{self.platform}>")
            cmake_cmd.append(f"-DTHINKER_TARGET_PLATFORM={target_platform}")
            
            cmake_cmd.append("..")
            subprocess.run(cmake_cmd, check=True)
            subprocess.run(["make", "-j16"], check=True)

    def run_thinker_inference(self, input_data, reference_input_data):
        if self.lib_path is None:
            self._build_thinker()
            self.lib_path = "executor/bin/libthinker.so"

        print("  ⚙️ -> ThinkerRunner start init.")
        thinker_runner = ThinkerRunner(self.lib_path,  self.platform, self.dynamic_shape)
        print("  ✅ -> ThinkerRunner init succeed.")

        print("  📦 -> ThinkerRunner start load model.")
        ret = thinker_runner.load_model(self.onnx_path, self.model_resource_path)
        if ret:
            print("  ✅ -> ThinkerRunner load model successfuly.")
        else:
            print("  ❌ -> ThinkerRunner failed to load model.")
            sys.exit(1)

        print("  🚀 -> ThinkerRunner start run.")
        thinker_runner.run(input_data, reference_input_data)
        print("  ✅ -> ThinkerRunner run successfuly.")

        thinker_runner.finalize()
        
    def validate(self):
        print(f"1️⃣ {Colors.BLUE} Starting generate inputs for linger and thinker.{Colors.RESET}")
        linger_input, thinker_input = self._generate_input()
        if self.dynamic_shape:
            self.onnx_model.specialize_dynamic_reshapes()
            reference_path = self.workspace_dir / "dynamic_reference.onnx"
            onnx.save(self.onnx_model.model, reference_path)
            self.reference_onnx_path = str(reference_path)
        print(f"✅{Colors.GREEN} All inputs have been generated successfuly.{Colors.RESET}")
        
        print(f"2️⃣ {Colors.BLUE} Linger onnxrunner inference start.{Colors.RESET}")
        self.run_linger_inference(linger_input)
        print(f"✅{Colors.GREEN} Linger onnxrunner inference succeed.{Colors.RESET}")

        print(f"3️⃣ {Colors.BLUE} Thinker inference start.{Colors.RESET}")
        self.run_thinker_inference(thinker_input, linger_input)
        print(f"✅{Colors.GREEN} ThinkerRunner inference succeed.{Colors.RESET}")

        cwd = Path.cwd()

        fixed_linger_dump_dir = cwd / "data" / "onnxrunner_int"
        fixed_thinker_dump_dir = cwd / "workspace" / "data"

        target_base_dir = cwd / "workspace" / self.onnx_name
        target_linger_dump_dir = target_base_dir / "dump_linger"
        target_thinker_dump_dir = target_base_dir / "dump_thinker"

        _move_dump_dir(fixed_linger_dump_dir, target_linger_dump_dir)
        _move_dump_dir(fixed_thinker_dump_dir, target_thinker_dump_dir)

        _remove_if_exists(fixed_linger_dump_dir.parent)
        _remove_if_exists(fixed_thinker_dump_dir)

        print(f"4️⃣ {Colors.BLUE} Consistency verification start.{Colors.RESET}")

        validator = Validator(self.onnx_model.model, str(target_linger_dump_dir), str(target_thinker_dump_dir), self.tensor_shapes)
        validator.compare()

def parse_dynamic_cfg(cfg_input) -> Dict:
    """
    Parse dynamic config.
    Supports input as a single string: "key=v1:v2,key2=v3:v4"
    """
    cfg: Dict = {}

    if not cfg_input:
        return cfg

    if isinstance(cfg_input, list):
        items = cfg_input
    else:
        items = [x.strip() for x in cfg_input.split(',') if x.strip()]

    for item in items:
        if '=' not in item:
            raise ValueError(
                f"Invalid --cfg format '{item}', expected key=value"
            )

        key, value_str = item.split('=', 1)
        key = key.strip()
        value_str = value_str.strip()

        if not key:
            raise ValueError(f"Empty key in --cfg '{item}'")

        try:
            values = tuple(int(v) for v in value_str.split(':'))
        except ValueError:
            raise ValueError(
                f"Invalid value for --cfg '{item}', only integer values separated by ':' are supported"
            )

        if len(values) != 3:
            raise ValueError(
                f"Invalid value for --cfg '{item}', expected min:max:factor"
            )
        min_value, max_value, factor = values
        if min_value <= 0 or max_value < min_value or factor <= 0:
            raise ValueError(
                f"Invalid value for --cfg '{item}', require 0 < min <= max and factor > 0"
            )
        first_valid = ((min_value + factor - 1) // factor) * factor
        if first_valid > max_value:
            raise ValueError(
                f"Invalid value for --cfg '{item}', no factor-aligned value exists in the range"
            )

        cfg[key] = values

    return cfg

def main():
    parser = argparse.ArgumentParser(
        description="Thinker Consistency Validation Tool", add_help=True)
    parser.add_argument('-g', '--onnx_path', type=str, required=True, help='Onnx model path (required).')
    parser.add_argument('-r', '--res_path', type=str, required=False, help='Model Resource path. Required for manual packaging.')
    parser.add_argument('-l', '--lib_path', type=str, required=False, help='Thinker dynamic library. Required when executed outside the project root directory.')
    parser.add_argument('-i', '--input_path', nargs='+', type=str, required=False, help='One or more input paths. Required when input is specified manually.')
    parser.add_argument('--cfg', type=str, default=None, help='Dynamic config in key=value format')
    parser.add_argument('--param_check', '--thinker_param_check', '--THINKER_PARAM_CHECK', dest='param_check', type=str, default='ON', choices=['ON', 'OFF', 'on', 'off', '1', '0', 'true', 'false', 'TRUE', 'FALSE'], help='Enable THINKER_PARAM_CHECK when building Thinker (default: ON).')
    parser.add_argument('--runtime_check', '--thinker_runtime_check', '--THINKER_RUNTIME_CHECK', dest='runtime_check', type=str, default='ON', choices=['ON', 'OFF', 'on', 'off', '1', '0', 'true', 'false', 'TRUE', 'FALSE'], help='Enable THINKER_RUNTIME_CHECK when building Thinker (default: ON).')
    
    args = parser.parse_args()

    dynamic_cfg = parse_dynamic_cfg(args.cfg)
    if len(dynamic_cfg) == 0:
        dynamic_cfg = None

    inputs: List[np.ndarray] = None
    if args.input_path is not None:
        inputs = []
        for path in args.input_path:
            inputs.append(np.load(path))
    
    validator = ThinkerValidator(
        onnx_path=args.onnx_path,
        model_resource_path=args.res_path,
        lib_path=args.lib_path,
        inputs=inputs,
        dynamic_cfg=dynamic_cfg,
        param_check=_cmake_bool(args.param_check) == "ON",
        runtime_check=_cmake_bool(args.runtime_check) == "ON"
    )
    validator.validate()

if __name__ == "__main__":
    main()

__all__ = ["Validator", "ThinkerValidator"]
