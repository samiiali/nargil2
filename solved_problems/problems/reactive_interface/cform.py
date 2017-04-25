#!/workspace/Library/Anaconda/install/bin/python

import sys
import os
import fileinput
import re

print "The python executable is: " + sys.executable
dirname = os.path.dirname(os.path.realpath(__file__))
os.chdir(dirname)

file = open("file1_mod.txt", "w")

for line in fileinput.input("file1.txt"):
    print line
    line = re.sub(r'Sin\(', r'sin(', line)
    line = re.sub(r'Sinh\(', r'sinh(', line)
    line = re.sub(r'Cos\(', r'cos(', line)
    line = re.sub(r'Cosh\(', r'cosh(', line)
    line = re.sub(r'Power\(E,', r'exp(', line)
    line = re.sub(r'Power\(', r'pow(', line)
    print line
    file.write(line)

file.close()