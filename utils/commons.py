import os

import numpy as np


# === COMMON FUNCTIONS ===
# == Make dir if neccessary
def checkOrMake(path):
    if not os.path.exists(path ):
        os.makedirs(path)

# == Remove '[' and ']' in a string
def arch2str(s):
    s = np.core.defchararray.replace(s, '[', '')
    s = np.core.defchararray.replace(s, ']', '')
    s = np.core.defchararray.replace(s, ' ', '')
    return str(s)