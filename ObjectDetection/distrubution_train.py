"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/12/30-22:08
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
import torch.distributed as dist

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'
dist.init_process_group(backend='gloo', init_method='env://', rank=0,
                        world_size=int(os.environ['WORLD_SIZE']
                                       ) if 'WORLD_SIZE' in os.environ else 1)