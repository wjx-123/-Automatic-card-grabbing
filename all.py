import os
import time
import subprocess
from pynvml import *

def check_gpus():
    # 初始化 NVML 库
    nvmlInit()

    # 获取 GPU 数量
    device_count = nvmlDeviceGetCount()
    free_gpus = []

    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        util_info = nvmlDeviceGetUtilizationRates(handle)

        # 检查 GPU 是否空闲（这里假定如果内存利用率非常低，则 GPU 为空闲）
        if util_info.gpu < 10:  # GPU 利用率小于 10% 视为空闲
            free_gpus.append(i)

    nvmlShutdown()
    return free_gpus

def occupy_gpu(gpu_id):
    # 占用指定 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # 运行一个持续占用 GPU 的进程（例如，一个训练脚本）
    subprocess.Popen(["python", "your_training_script.py"])

if __name__ == "__main__":
    while True:
        free_gpus = check_gpus()
        if free_gpus:
            print("空闲的 GPU：", free_gpus)
            for gpu in free_gpus:
                occupy_gpu(gpu)
        else:
            print("没有空闲的 GPU，等待中...")
        time.sleep(60)  # 每 60 秒检查一次