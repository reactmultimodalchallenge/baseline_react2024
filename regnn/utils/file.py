import os

def mkdir_if_not_exist(pth):
    if not os.path.isdir(pth):
        os.makedirs(pth)