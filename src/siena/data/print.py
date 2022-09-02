import os
import numpy as np
from textwrap import wrap

with open('logo.txt') as f:
    logo = f.readlines()

terminalsize = os.get_terminal_size()
logosize = (len(logo[0]),len(logo)) #(x,y) assume rectengular shape

for i in logo:
    line = i.rsplit('\n')[0]
    string = ("%s".center((terminalsize[0]-logosize[0]//2) //2) % line)[:terminalsize[0]]
    print(string)
