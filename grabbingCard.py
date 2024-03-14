import torch
from pynvml import *
import time
import sys
import subprocess

# 初始化NVML，用于获取GPU状态信息
nvmlInit()

def select_gpu(count=torch.cuda.device_count(), threshold=1024, second=5):
    if count == 0:
        return 'cpu'
    current = 0
    while True:
        handle = nvmlDeviceGetHandleByIndex(current)
        info = nvmlDeviceGetMemoryInfo(handle)
        used_memory = info.used // (1024 * 1024)  # 将字节转换为MB
        if used_memory < threshold:
            sys.stderr.write(f'GPU{current}空闲（使用内存{used_memory}MB），低于阈值{threshold}MB。选择GPU{current}。\n')
            nvmlShutdown()  # 完成NVML操作后关闭
            return current
        else:
            sys.stderr.write(f'GPU{current}正在使用（使用内存{used_memory}MB），继续检查下一个GPU。\n')
        time.sleep(second)
        current = (current + 1) % count

if __name__ == '__main__':
    sys.stderr.write('程序开始运行。\n')
    device = select_gpu(threshold=3000)  # 调整threshold根据需要
    if isinstance(device, int):  # 成功选择一个GPU
        # 构建并运行train_unconditional.py的命令，将GPU索引作为命令行参数传递
        command = f'python train_unconditional.py --device {device}'
        try:
            subprocess.run(command, check=True, shell=True)
            sys.stderr.write('train_unconditional.py运行结束。\n')
        except subprocess.CalledProcessError as e:
            sys.stderr.write(f'运行train_unconditional.py时出错：{e}\n')
    else:
        sys.stderr.write('没有找到符合条件的GPU。\n')