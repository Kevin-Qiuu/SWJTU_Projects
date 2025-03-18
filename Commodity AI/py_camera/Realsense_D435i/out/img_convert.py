import glob
import os
from PIL import Image

for i in glob.glob('F:\\pstest\\*.bmp', recursive=True):

    img = Image.open(i)
    path_name = os.path.split(i)[0]
    ##换一个后缀
    path_cat = os.path.split(i)[1].replace("bmp","jpg")
    new_path = os.path.join(path_name,path_cat)
    img.save(new_path,quality = 99)
