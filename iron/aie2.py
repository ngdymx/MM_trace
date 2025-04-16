import argparse

from ml_dtypes import bfloat16
import numpy as np
import sys
from aie.extras.dialects.ext import arith

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
import aie.utils.trace as trace_utils
from aie.utils.trace import PortEvent
from aie.utils.trace_events_enum import CoreEvent
from aie.helpers.dialects.ext.scf import _for as range_

K = 256
N = 128
trace_size = 131072

def MM():
    @device(AIEDevice.npu2)
    def device_body():

        a_ty = np.ndarray[(8, N), np.dtype[bfloat16]]
        b_ty = np.ndarray[(N, 16), np.dtype[bfloat16]]
        o_ty = np.ndarray[(8, 16), np.dtype[bfloat16]]

        # AIE Core Function declarations
        mm = external_func("mm", inputs=[a_ty, b_ty, o_ty])

        # Tile declarations
        shim_tile0 = tile(0, 0)
        mem_tile0 = tile(0, 1)
        compute_tile02 = tile(0, 2)

        # AIE-array data movement with object fifos
        # Input a
        mem_a_fifo = object_fifo("mem_in_a", shim_tile0, mem_tile0, 2 , a_ty)
        in_a_fifo = object_fifo("in_a_fifo", mem_tile0, compute_tile02, 2, a_ty, ([(1, 8 * N),(N // 8, 8),(8, N),(8, 1)]))
        object_fifo_link(mem_a_fifo, in_a_fifo, [], [])
        # Input b
        mem_b_fifo = object_fifo("mem_in_b", shim_tile0, mem_tile0, 2, b_ty)
        in_b_fifo = object_fifo("in_b_fifo", mem_tile0, compute_tile02, 2, b_ty, ([(N // 8, 8 * 16), (16 // 8, 8),(8, 16),(8, 1)]))
        object_fifo_link(mem_b_fifo, in_b_fifo, [], [])

        # Out o
        out_o_fifo = object_fifo("out_o_fifo", compute_tile02, shim_tile0, 2, o_ty, ([(8, 8), (16 // 8, 64), (8, 1)]))

        @core(compute_tile02, "mm.o")
        def core_body():
            for _ in range_(sys.maxsize):

                elem_in_a = in_a_fifo.acquire(ObjectFifoPort.Consume, 1)
                for _ in range_(K // 16):
                    elem_out = out_o_fifo.acquire(ObjectFifoPort.Produce, 1)
                    elem_in_b = in_b_fifo.acquire(ObjectFifoPort.Consume, 1)
                    mm(elem_in_a, elem_in_b, elem_out)
                    in_b_fifo.release(ObjectFifoPort.Consume, 1)
                    out_o_fifo.release(ObjectFifoPort.Produce, 1)
                in_a_fifo.release(ObjectFifoPort.Consume, 1)

        tiles_to_trace = [compute_tile02, shim_tile0]
        trace_utils.configure_packet_tracing_flow(tiles_to_trace, shim_tile0)

        # To/from AIE-array data movement
        @runtime_sequence(
            np.ndarray[(8, N), np.dtype[bfloat16]],
            np.ndarray[(N, K), np.dtype[bfloat16]],
            np.ndarray[(8, K), np.dtype[bfloat16]]
        )

        def sequence(a, b, o):
            trace_utils.configure_packet_tracing_aie2(
                tiles_to_trace = tiles_to_trace,
                shim = shim_tile0,
                trace_size = trace_size,
                trace_offset = 0,
                coretile_events = [
                    PortEvent(CoreEvent.PORT_RUNNING_0, 1, True),  # master(1)
                    PortEvent(CoreEvent.PORT_RUNNING_2, 2, True),  # master(0)
                    PortEvent(CoreEvent.PORT_RUNNING_1, 1, False), # slave(1)
                    CoreEvent.INSTR_EVENT_0,
                    CoreEvent.INSTR_EVENT_1,
                    CoreEvent.INSTR_VECTOR,
                    #CoreEvent.INSTR_LOAD,
                    CoreEvent.INSTR_STORE,
                    #CoreEvent.INSTR_CALL,
                    CoreEvent.MEMORY_STALL,
                ]
            )
            npu_dma_memcpy_nd(
                metadata = mem_a_fifo,
                bd_id=0,
                mem=a,
                sizes=[1, 1, 1, 8 * N],
                strides=[0, 0, 0, 1],
            )
            npu_dma_memcpy_nd(
                metadata=mem_b_fifo,
                bd_id=1,
                mem=b,
                sizes=[1, K // 16, N, 16],
                strides=[0, 16, K, 1],
            )
            npu_dma_memcpy_nd(
                metadata=out_o_fifo,
                bd_id=2,
                mem=o,
                sizes=[1, K // 16, 8, 16],
                strides=[0, 16, K, 1],
            )
            dma_wait(out_o_fifo)
            trace_utils.gen_trace_done_aie2(shim_tile0)

# Declares that subsequent code is in mlir-aie context
with mlir_mod_ctx() as ctx:
    MM()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)

