import inspect
import sys
import os

import torch
import random
import numpy as np

def raiseNotDefined():
    filename = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]
    
    print(f"*** Method not implemented: {method} at line {line} of {filename} ***")
    sys.exit()
    
def seed_everything(seed, env):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    env.seed(seed)
    
YOUR_CODE_HERE = "*** YOUR CODE HERE ***"