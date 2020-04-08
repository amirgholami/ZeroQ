#!/usr/bin/python
#!/usr/bin/python3

# This script assume exclusive usage of the GPUs.
# If you have limited usage of GPUs, you can limit the range of gpu indices you are using.

import threading
import time
import os
import numpy as np

import gpustat
import logging

import itertools

FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] %(message)s'
FORMAT_MINIMAL = '%(message)s'

logger = logging.getLogger('runner')
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)

exitFlag = 0
GPU_MEMORY_THRESHOLD = 1000  # MB?


def get_free_gpu_indices():
    '''
        Return an available GPU index.
    '''
    while True:
        stats = gpustat.GPUStatCollection.new_query()
        # print('stats length: ', len(stats))
        return_list = []
        for i, stat in enumerate(stats.gpus):
            memory_used = stat['memory.used']
            if memory_used < GPU_MEMORY_THRESHOLD:  # and i in [2,3,4,5,6,7]:
                return i

        logger.info("Waiting on GPUs")
        time.sleep(10)


class DispatchThread(threading.Thread):
    def __init__(self, threadID, name, counter, bash_command_list):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.bash_command_list = bash_command_list

    def run(self):
        logger.info("Starting " + self.name)
        # print_time(self.name, self.counter, 5)
        threads = []
        for i, bash_command in enumerate(self.bash_command_list):

            cuda_device = get_free_gpu_indices()
            thread1 = ChildThread(1, f"{i}th + {bash_command}", 1, cuda_device,
                                  bash_command)
            thread1.start()
            import time
            time.sleep(10)
            threads.append(thread1)

        # join all.
        for t in threads:
            t.join()
        logger.info("Exiting " + self.name)


class ChildThread(threading.Thread):
    def __init__(self, threadID, name, counter, cuda_device, bash_command):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.cuda_device = cuda_device
        self.bash_command = bash_command

    def run(self):
        # os.environ['CUDA_VISIBLE_DEVICES'] = f'{self.cuda_device[0]},{self.cuda_device[1]}'
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{self.cuda_device}'
        bash_command = self.bash_command

        logger.info(f'executing {bash_command} on GPU: {self.cuda_device}')
        # ACTIVATE
        os.system(bash_command)
        import time
        import random
        time.sleep(random.random() % 5)
        logger.info("Finishing " + self.name)


#####################
# this is for model training on CIFAR-10
#####################
BASH_COMMAND_LIST = []

for run_id in range(64):
    comm = f"python  refined_gaussian.py --model=resnet18 --work_dir=new_logs/data/resnet18_refined_uniform --batch_size=128 --num_batch=1 --fit_BNs=20 --id={run_id} --init=uniform"
    # comm = f'python test.py --model=resnet18 --test_batch_size=1024 --act_bit=8 --data=refined --fit_BNs=20 --work_dir=logs/refined/ --act_range=refined --id={run_id}'
    BASH_COMMAND_LIST.append(comm)

# Create new threads
dispatch_thread = DispatchThread(2, "Thread-2", 8, BASH_COMMAND_LIST)

# Start new Threads
dispatch_thread.start()
dispatch_thread.join()

import time
time.sleep(5)

logger.info("Exiting Main Thread")
