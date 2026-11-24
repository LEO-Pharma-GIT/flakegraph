# SPDX-License-Identifier: Apache-2.0
"""Loaded by Python at start in a pooling server's container (on PYTHONPATH).

On a unified-memory GPU (DGX Spark, GH200) vLLM takes the kernel's
MemAvailable as the device's free memory and refuses to start when its share
is larger. A node tuned to keep memory truly free for CUDA - a high
vm.watermark_scale_factor, because the CUDA driver does not reclaim page
cache - shows a MemAvailable lowered by the very reserve it keeps, so a server
whose share fits in that free memory is refused. With
FLAKEGRAPH_UMA_FREE_MEMORY=free the server counts MemFree instead: what
cudaMemGetInfo reports there, and what an allocation can take without
reclaim. Nothing changes when the variable is unset.
"""

import os

if os.environ.get("FLAKEGRAPH_UMA_FREE_MEMORY") == "free":
    import psutil

    _virtual_memory = psutil.virtual_memory

    def _free_as_available():  # type: ignore[no-untyped-def]
        memory = _virtual_memory()
        return memory._replace(available=memory.free)

    psutil.virtual_memory = _free_as_available
