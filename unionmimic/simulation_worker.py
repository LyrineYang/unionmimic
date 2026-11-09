"""Process-owned Isaac Gym lifetime; no Python/native destructor race at exit."""
import faulthandler
import os
import sys
import traceback


def run_isolated_simulation(function,args):
    # These commands own one simulator and exit after their HDF5/JSON context
    # managers have committed. The OS releases their CUDA/Vulkan contexts.
    # Normal interpreter finalizers can race the old native simulator workers.
    os.environ['UNIONMIMIC_PROCESS_OWNS_SIM']='1'
    faulthandler.enable(all_threads=True)
    code=1
    try:
        function(args)
        code=0
    except BaseException:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
