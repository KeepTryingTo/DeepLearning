"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/1/4 14:18
"""

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from models.detr import build
# from .detr_lite import build

def build_model(args):
    return build(args)