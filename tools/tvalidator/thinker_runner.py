from .thinker_bind import bind_functions, tData, tMemory, tModelHandle, tExecHandle, dtype_bytes
from .thinker_bind import DTYPE_TO_NP
import ctypes as ct
from ctypes import (
    Structure, POINTER, byref,
    c_int, c_uint64, c_int8, c_size_t, c_void_p,
    CDLL, create_string_buffer, addressof, cast
)
import os
import subprocess
from pathlib import Path
import numpy as np

class ThinkerRunner:
    def __init__(self, lib_path="./bin/libthinker.so", platform = 'venusA', dynamic_shape = False):
        """
        lib_path: Path of libthinker.so
        platform: Platform to run, e.g. venus, arcs, venusA
        """
        try:
            self.lib = ct.CDLL(lib_path)
            bind_functions(self.lib)
            self.lib.tInitialize()
        except OSError as e:
            print(f"Error: Failed to load library at {lib_path}. {e}")
            self.lib = None
        
        self.platform = platform
        self.dynamic_shape = dynamic_shape
        self.psram_buffer = None
        self.psram_size = 0
        self.share_buffer = None
        self.share_size = 0
        self.res_buffer = None
        self.res_size = 0
        self.use_psram_size = 0  
        self.use_share_size = 0

        self.model_hdl = None
        self.exec_hdl = None

        self._init_dump_dir()
        self._init_platform()

    def _init_dump_dir(self):
        dump_dir = 'workspace/data'
        if os.path.exists(dump_dir):
            return
        else:
            Path(dump_dir).mkdir(parents=True, exist_ok=True)


    def _init_platform(self):
        if self.platform == "venusA":
            try:
                self.psram_size = 8*1024*1024
                self.psram_buffer = ct.create_string_buffer(self.psram_size)
            except MemoryError:
                print(f"Error: Failed to allocate {self.psram_size} bytes for PSRAM.")
            
            try:
                self.share_size = 384*1024
                self.share_buffer = ct.create_string_buffer(self.share_size)
            except MemoryError:
                print(f"Error: Failed to allocate {self.share_size} bytes for PSRAM.")

        elif self.platform == "arcs":
            try:
                self.psram_size = 8*1024*1024
                self.psram_buffer = ct.create_string_buffer(self.psram_size)
            except MemoryError:
                print(f"Error: Failed to allocate {self.psram_size} bytes for PSRAM.")
            
            try:
                self.share_size = 384*1024
                self.share_buffer = ct.create_string_buffer(self.share_size)
            except MemoryError:
                print(f"Error: Failed to allocate {self.share_size} bytes for PSRAM.")

        elif self.platform == "venus":
            try:
                self.psram_size = 8*1024*1024
                self.psram_buffer = ct.create_string_buffer(self.psram_size)
            except MemoryError:
                print(f"Error: Failed to allocate {self.psram_size} bytes for PSRAM.")
            
            try:
                self.share_size = 384*1024
                self.share_buffer = ct.create_string_buffer(self.share_size)
            except MemoryError:
                print(f"Error: Failed to allocate {self.share_size} bytes for PSRAM.")

    def load_model(self, onnx_path = None, thinker_res_path = None):
        assert onnx_path is not None or thinker_res_path is not None, "Either `onnx_path` or `thinker_res_path` must be provided."
        model_file_to_load = None
        
        if thinker_res_path is None:
            thinker_res_dir = "data.ignore"
            if os.path.exists(thinker_res_dir):
                subprocess.run(f"rm -rf {thinker_res_dir}", shell=True)
            Path(thinker_res_dir).mkdir(parents=True, exist_ok=True)
            
            thinker_res_path = f"{thinker_res_dir}/test.bin"
            cmd = f"tpacker -g {onnx_path} -d True -r 393216 --dma_prefetch=True -o {thinker_res_path}"
            
            print(f"  ->-> Running tpacker: {cmd}")
            exit_code = os.system(cmd)
            
            if exit_code != 0:
                print(f"  ->-> Error: tpacker command failed with code {exit_code}")
                return False
                
            model_file_to_load = thinker_res_path
            
        else:
            if not os.path.isabs(thinker_res_path):
                model_file_to_load = os.path.join(os.getcwd(), thinker_res_path)
            else:
                model_file_to_load = thinker_res_path
        
        if not model_file_to_load or not os.path.exists(model_file_to_load):
            print(f"    ->-> Error: Model resource file not found at {model_file_to_load}")
            return False

        try:
            print(f"    ->-> Reading model resource file: {model_file_to_load}")
            with open(model_file_to_load, 'rb') as f:
                file_data = f.read()
        except IOError as e:
            print(f"    ->-> Error reading file {model_file_to_load}: {e}")
            return False

        self.res_size = len(file_data)
        if self.res_size == 0:
            print(f"    ->-> Warning: Model resource file {model_file_to_load} is empty.")
            
        self.res_buffer = ct.create_string_buffer(file_data, self.res_size)
        
        print(f"    ->-> Successfully loaded {self.res_size} bytes into internal buffer.")
        print(f"    ->-> Buffer address: {ct.addressof(self.res_buffer)}")
        
        return True

    def run(self, input_list):
        # import pdb; pdb.set_trace()
        T_SUCCESS = 0

        num_memory = ct.c_int32(0)
        MAX_MEM_BLOCKS = 7
        memory_list = (tMemory * MAX_MEM_BLOCKS)()

        ret = self.lib.tGetMemoryPlan(
            memory_list,          # POINTER(tMemory)
            ct.byref(num_memory), # POINTER(ct.c_int32)
            ct.cast(self.res_buffer, ct.POINTER(ct.c_byte)),   # POINTER(ct.c_int8)
            self.res_size              # ct.c_uint64
        )
        if ret != T_SUCCESS:
            raise RuntimeError(f"tGetMemoryPlan failed with code: {ret}")
        
        g_psram_buf_addr = ct.c_uint64(ct.addressof(self.psram_buffer))
        g_share_buf_addr = ct.c_uint64(ct.addressof(self.share_buffer))
        PSRAM_BUFFER_SIZE = self.psram_size
        SHARE_BUFFER_SIZE = self.share_size
         

        for i in range(num_memory.value):
            mem_size = memory_list[i].size_
        
            if memory_list[i].dptr_ == 0:
                aligned_size = (mem_size + 63) & (~63)
                if memory_list[i].dev_type_ == 1 or memory_list[i].dev_type_ == 3:
                    if self.use_psram_size + aligned_size > PSRAM_BUFFER_SIZE:
                        raise MemoryError(f"PSRAM buffer overflow: "
                                          f"Required {self.use_psram_size + aligned_size}, "
                                          f"but only {PSRAM_BUFFER_SIZE} available.")
                    
                    memory_list[i].dptr_ = g_psram_buf_addr.value + self.use_psram_size
                    self.use_psram_size += aligned_size
                elif memory_list[i].dev_type_ == 2:
                    if self.use_share_size + aligned_size > SHARE_BUFFER_SIZE:
                        raise MemoryError(f"Share buffer overflow: "
                                          f"Required {self.use_share_size + aligned_size}, "
                                          f"but only {SHARE_BUFFER_SIZE} available.")

                    memory_list[i].dptr_ = g_share_buf_addr.value + self.use_share_size
                    self.use_share_size += aligned_size
        
        print("    ->-> Buffer allocation successful.")
        print(f"    ->-> PSRAM used: {self.use_psram_size} / {PSRAM_BUFFER_SIZE}")
        print(f"    ->-> Share used: {self.use_share_size} / {SHARE_BUFFER_SIZE}")

        model_hdl = tModelHandle(0)
        print("    ->-> Calling tModelInit")
        ret = self.lib.tModelInit(
            ct.byref(model_hdl),
            ct.cast(self.res_buffer, ct.POINTER(ct.c_byte)),
            self.res_size,
            memory_list,
            num_memory
        )
        if ret != T_SUCCESS:
            print(f"    ->-> tInitModel: ret = {ret}")
            raise RuntimeError(f"   ->-> tModelInit failed with code: {ret}")
        else:
            self.model_hdl = model_hdl
        
        hdl = tExecHandle(0)
        print("    ->-> Calling tCreateExecutor...")
        ret = self.lib.tCreateExecutor(
            model_hdl,
            ct.byref(hdl),
            memory_list,
            num_memory
        )
        
        if ret != T_SUCCESS:
            print(f"    ->-> tCreateExecutor: ret = {ret}")
            raise RuntimeError(f"tCreateExecutor failed with code: {ret}")
        else:
            self.exec_hdl = hdl

        print(f"    ->-> Model and executor created successfully for this run (Handle: {hdl.value})")

        input_count = self.lib.tGetInputCount(model_hdl)
        print(f"    ->-> Model requires {input_count} inputs.")
        assert len(input_list) == input_count, \
        f"Input mismatch: Model needs {input_count} inputs, but {len(input_list)} were provided."

        self._input_info_list = []
        for i in range(input_count):
            input_array = np.ascontiguousarray(input_list[i])
            
            input_info = tData()
            ret = self.lib.tGetInputInfo(hdl, i, ct.byref(input_info))
            if ret != T_SUCCESS:
                raise RuntimeError(f"tGetInputInfo failed for input {i}: {ret}")
            
            if self.dynamic_shape:
                for j in range(len(input_array.shape)):
                    input_info.shape_.dims_[j] = input_array.shape[j]
                    
            input_info.dptr_ = input_array.ctypes.data_as(ct.c_void_p)
            
            ret = self.lib.tSetInput(hdl, i, ct.byref(input_info))
            if ret != T_SUCCESS:
                raise RuntimeError(f"tSetInput failed for input {i}: {ret}")
                
            self._input_info_list.append(input_info) # keep lifetime of input_info
        print("    ->-> All inputs set successfully.")

        if self.dynamic_shape:
            print("    ->-> Running tUpdateShape.")
            ret = self.lib.tUpdateShape(self.exec_hdl, None, None, 0)
            if ret != T_SUCCESS:
                print(f"tUpdateShape: ret = {ret}")
                raise RuntimeError(f"tUpdateShape failed with code: {ret}")
            print("    ->-> tUpdateShape complete.")

        print("    ->-> Running tForward")
        ret = self.lib.tForward(hdl)
        if ret != T_SUCCESS:
            print(f"tForward: ret = {ret}")
            raise RuntimeError(f"tForward failed with code: {ret}")
        print("    ->-> tForward complete.")


    def get_output(self):
        output_count = self.lib.tGetOutputCount(self.model_hdl)
        print(f"    ->-> Model generates {output_count} outputs.")

        out_arrays = []

        self._output_info_list = []
        for i in range(output_count):
            print(f"     ->->-> Start getting [Output {i}]:")
            output_info = tData()
            ret = self.lib.tGetOutput(self.exec_hdl, i, ct.byref(output_info))
            if ret != 0:
                raise RuntimeError(f"tGetOutput failed at index {i}, ret={ret}")
            self._output_info_list.append(output_info)

            tensor_elems = 1
            for j in range(output_info.shape_.ndim_):
                dim = output_info.shape_.dims_[j]
                if dim == 0:
                    break
                tensor_elems *= dim
            elem_size = dtype_bytes(output_info.dtype_)
            tensor_bytes = tensor_elems * elem_size

            np_dtype = DTYPE_TO_NP.get(output_info.dtype_, np.float32)

            dst_addr = ct.addressof(self.psram_buffer) + self.use_psram_size
            dst_ptr = ct.c_void_p(dst_addr)

            ct.memmove(dst_ptr, output_info.dptr_, tensor_bytes)

            dims = [output_info.shape_.dims_[i] for i in range(output_info.shape_.ndim_) if output_info.shape_.dims_[i] > 0]
            print(f"    ->->-> [Output {i}] shape={dims}, dtype={np_dtype.__name__}, bytes={tensor_bytes}")
            np_arr = np.frombuffer((ct.c_char * tensor_bytes).from_address(dst_addr), dtype=np_dtype).reshape(dims)

            self.use_psram_size += (tensor_bytes + 63) & (~63)

            out_arrays.append(np_arr)
        
        return out_arrays


    def finalize(self):
        self.lib.tUninitialize()
