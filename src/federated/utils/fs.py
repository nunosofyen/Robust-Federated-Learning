import shutil 
import os 
import math

def delete_folder(fd):
    shutil.rmtree(fd)

def create_folder(fd):
    if not os.path.exists(fd):
        print(f"\nCREATING FOLDER {fd} \n")
        os.makedirs(fd)

def create_file(filename):
    print(f"\nCREATING FILE {filename} \n")
    with open(filename, "w") as file:
        file.close()

def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n