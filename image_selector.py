import time
import subprocess
import shutil
import msvcrt
from os import listdir
from os.path import isfile, join
import random


unclass_path = "bark_unclassified/"
low_path = "bark_dataset/low/"
mod_path = "bark_dataset/moderate/"
high_path = "bark_dataset/high/"

only_files = [f for f in listdir(unclass_path) if isfile(join(unclass_path, f))]

random.shuffle(only_files)

for file in only_files:

    cmd = 'ImageViewer -o on ' + unclass_path + file

    print(cmd)
    p = subprocess.Popen(cmd)
    print("Enter classification [L, M, H]")
    x = str(msvcrt.getch(),'utf-8')
    print(x)
    p.kill()

    if x.upper() == 'L':
        print("moving to low_fuel_load")
        shutil.move(unclass_path + file,low_path + file)
    elif x.upper() == 'M':
        print("moving to moderate_fuel_load")
        shutil.move(unclass_path + file,mod_path + file)
    elif x.upper() == 'H':
        print("moving to high_fuel_load")
        shutil.move(unclass_path + file,high_path + file)
    elif x.upper() == 'E':
        break
    
   
    

