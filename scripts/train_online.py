#!/usr/bin/env python
"""Dedicated native worker: flush files, propagate errors, then exit atomically."""
import ctypes
import os
import sys
import traceback
ctypes.CDLL('libEGL_nvidia.so.0',mode=os.RTLD_NOW|os.RTLD_LOCAL|os.RTLD_NODELETE)
import isaacgym  # Must precede torch, including imports through the learner.
import torch
from unionmimic.online.worker import main

if __name__=='__main__':
    code=0
    try:
        main()
    except BaseException:
        traceback.print_exc();code=1
    finally:
        if torch.cuda.is_initialized():torch.cuda.synchronize()
        sys.stdout.flush();sys.stderr.flush()
    # Isaac Gym's known exit-time native finalizer crash is isolated here.
    # All task exceptions retain a nonzero exit code and a failed status.
    os._exit(code)
