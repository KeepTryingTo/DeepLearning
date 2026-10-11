
import os
import sys
import json
import random
from ast import literal_eval

import numpy as np
import torch

# -----------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logging(config):
    """ monotonous bookkeeping """
    work_dir = config.system.work_dir
    # create the work directory if it doesn't already exist
    os.makedirs(work_dir, exist_ok=True)
    # log the args (if any)
    with open(os.path.join(work_dir, 'args.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
    # log the config itself
    with open(os.path.join(work_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(config.to_dict(), indent=4))

class CfgNode:
    """ a lightweight configuration class inspired by yacs """
    # TODO: convert to subclass from a dict like in yacs?
    # TODO: implement freezing to prevent shooting of own foot
    # TODO: additional existence/override checks when reading/writing params?

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs) # 将传入的关键字参数转为类属性

    def __str__(self):
        #TODO 美化打印
        return self._str_helper(0)

    def _str_helper(self, indent):
        """ 需要一个助手来支持嵌套缩进，以实现美观的打印 """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """ return a dict representation of the config """
        return { k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items() }

    def merge_from_dict(self, d):
        self.__dict__.update(d)  # 将传入的关键字参数转为类属性

    def merge_from_args(self, args):
        """
        update the configuration from a list of strings that is expected to come from the command line, i.e. sys.argv[1:].
        从预期来自命令行的字符串列表更新配置，例如sys.argv[1:]。

        The arguments are expected to be in the form of `--arg=value`, and the arg can use . to denote nested sub-attributes. Example:
        参数应该是‘——arg=value ’的形式，并且该参数可以使用。表示嵌套的子属性。

        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:
            #TODO 获得参数名称以及参数设置的值
            keyval = arg.split('=')
            assert len(keyval) == 2, "expecting each override arg to be of form --arg=value, got %s" % arg
            key, val = keyval # unpack

            # first translate val into a python object
            try:
                """
                    安全地计算表达式节点或包含Python表达式的字符串。
                    所提供的字符串或节点只能由以下Python文字结构组成：
                    字符串、字节、数字、元组、列表、字典、集合、布尔值和None。
                """
                val = literal_eval(val)
                """
                need some explanation here.
                - if val is simply a string, literal_eval will throw a ValueError
                - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created
                """
            except ValueError:
                pass

            #TODO 判断参数名称的设置是否合法 find the appropriate object to insert the attribute into
            assert key[:2] == '--'
            key = key[2:] # TODO strip the '--'
            keys = key.split('.')#TODO 比如我们设置的属性是分层一级一级的
            """
            model.backbone.channel...类似这样的设置参数名称
            """
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k) #TODO 逐个获得子对象
            leaf_key = keys[-1] #TODO 最末级属性名称

            # TODO ensure that this attribute exists
            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"

            # TODO 修改最终属性值 overwrite the attribute
            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)
