import subprocess
import glob

# This script will clear the output of all jupyter notebooks in the current directory
for file in glob.glob("*/*.ipynb"):
    print(file, "is a jupyter notebook file")
    clearfile = subprocess.run(["jupyter", "nbconvert",  "--clear-output", "--inplace", file]) 
    print("The exit code of " , file, " is: " , clearfile.returncode)