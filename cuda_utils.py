import os
import time
import datetime
import numpy as np
import GPUtil

def get_device_name():
    gpus = GPUtil.getGPUs()
    return gpus[0].name

def dynamic_cuda_allocation():
    gpus = GPUtil.getGPUs()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax([gpu.memoryFree for gpu in gpus]))

def block_until_cuda_memory_free(required_mem, interval=30):
    start_time = time.time()
    gpus = GPUtil.getGPUs()
    available_mem = max([gpu.memoryFree for gpu in gpus])
    while available_mem < required_mem:
        blocked_time = str(datetime.timedelta(seconds=int(time.time()-start_time)))
        print(f"{time.ctime()} \t {available_mem} MiB < {required_mem} MiB : blocked for {blocked_time}", end="\r")
        time.sleep(interval)
        gpus = GPUtil.getGPUs()
        available_mem = max([gpu.memoryFree for gpu in gpus])
    blocked_time = str(datetime.timedelta(seconds=int(time.time()-start_time)))
    print(f"{time.ctime()} \t {available_mem} MiB >= {required_mem} MiB : passed after {blocked_time}", end="\r")