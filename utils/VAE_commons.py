
import os

def checkOrMake(path, dirID):
    if not os.path.exists(path + '/' + dirID):
        os.makedirs(path + '/' + dirID)
        return True
    else:
        return False