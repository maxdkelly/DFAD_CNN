import time
import subprocess
import shutil
from os import listdir
from os.path import isfile, join



unclass_path = "bark_unclassified/"
low_path = "bark_dataset/low/"
mod_path = "bark_dataset/moderate/"
high_path = "bark_dataset/high/"

only_files = [f for f in listdir(unclass_path) if isfile(join(unclass_path, f))]

for file in only_files:

    cmd = 'ImageViewer ' + unclass_path + file
    print(cmd)
    p = subprocess.Popen(cmd)
    x = input("Enter classification [L, M, H]")
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
    
   
    

