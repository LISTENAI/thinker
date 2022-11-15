# Copyright (C) 2022 listenai Co.Ltd
# All rights reserved. 
# Created by leifang on 2022.09.31

from typing import Dict
from .enum_defines import MemType, ALIGN16


def build_memory_plan_report(memory_plan: Dict):
    fp = open("memory.txt", "w")
    fp.write("=" * 126 + "\n")
    fp.write("|" + "*" * 52 + " memory plan result " + "*" * 52 + "|" + "\n")
    fp.write("=" * 126 + "\n")

    fp.write("|" + "-" * 124 + "|\n")

    memory_list = []
    for ctx in memory_plan.entry_ctx_list:
        tensor = ctx.entry.tensor
        if tensor is not None and tensor.data is not None:
            if tensor.mem_type not in memory_list:
                memory_list.append(tensor.mem_type)

    for d in range(len(memory_list)):
        params_buff = b""
        for ctx in memory_plan.entry_ctx_list:
            tensor = ctx.entry.tensor
            if (
                tensor is not None
                and tensor.data is not None
                and tensor.mem_type == memory_list[d]
            ):
                t = tensor.data
                offset = ALIGN16(len(t.tobytes()))
                params_buff += t.tobytes() + b"\0" * (offset - len(t.tobytes()))

        fp.write(
            "|{:<48} total size:{:<64}|\n".format(memory_list[d], len(params_buff))
        )
        fp.write("|" + "-" * 124 + "|\n")
        fp.write("|{}({})".format(len(params_buff), -1) + " " * 3 + "|\n")
        fp.write("|" + "-" * 124 + "|\n")
        fp.write(
            "|tensor name"
            + " " * 40
            + "mem id"
            + " " * 10
            + "life begin"
            + " " * 6
            + "life end"
            + " " * 13
            + "tensor size"
            + " " * 9
            + "|\n"
        )
        fp.write("|" + "-" * 124 + "|" + "\n")

        for i, ctx in enumerate(memory_plan.entry_ctx_list):
            tensor = ctx.entry.tensor
            if (
                tensor is not None
                and tensor.data is not None
                and tensor.mem_type == memory_list[d]
            ):
                if ctx.life_end == 100000000000:
                    fp.write(
                        "|{:<50} {:<15} {:<15} {:<20} {:<20}|\n".format(
                            ctx.entry.name,
                            ctx.mem_id,
                            ctx.life_begin,
                            "INF",
                            ctx.nbytes,
                        )
                    )
                else:
                    fp.write(
                        "|{:<50} {:<15} {:<15} {:<20} {:<20}|\n".format(
                            ctx.entry.name,
                            ctx.mem_id,
                            ctx.life_begin,
                            ctx.life_end,
                            ctx.nbytes,
                        )
                    )
        fp.write("|" + "-" * 124 + "|" + "\n\n")

    for k, v in memory_plan.mem_sizes.items():
        runtime_size = sum(v)
        fp.write("|" + "-" * 124 + "|" + "\n")
        fp.write(
            "|MemType.{:<50} total size:{:<54}|\n".format(MemType(k).name, runtime_size)
        )
        fp.write("|" + "-" * 124 + "|\n")
        for i in range(len(v)):
            fp.write("|{}({})".format(v[i], i) + " " * 3)
        fp.write("|\n")

        fp.write("|" + "-" * 124 + "|" + "\n")
        fp.write(
            "|tensor name"
            + " " * 40
            + "mem id"
            + " " * 10
            + "life begin"
            + " " * 6
            + "life end"
            + " " * 13
            + "tensor size"
            + " " * 9
            + "|\n"
        )
        fp.write("|" + "-" * 124 + "|" + "\n")

        for i, ctx in enumerate(memory_plan.entry_ctx_list):
            tensor = ctx.entry.tensor
            if tensor.mem_type.value == k and tensor.data is None:
                fp.write(
                    "|{:<50} {:<15} {:<15} {:<20} {:<20}|\n".format(
                        ctx.entry.name,
                        ctx.mem_id,
                        ctx.life_begin,
                        ctx.life_end,
                        ctx.nbytes,
                    )
                )
        fp.write("|" + "-" * 124 + "|" + "\n")

    fp.close()
    print("memory.txt generated success!")


__all__ = ["build_memory_plan_report"]
