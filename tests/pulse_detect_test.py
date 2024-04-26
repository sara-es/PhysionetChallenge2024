# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 22:13:25 2024

@author: hssdwo
"""
import numpy as np
import matplotlib.pyplot as plt


# initial ideas for trying to locate the reference pulse - assumptions:
#    - image is rotated correctly
#    - signal has been digitized correctly
#    - reference pulse should be one large gridline thick
#    - assume that large gridlines are between 25 and 40 pixels

# Question - is the bottom of the gridline 0mV?

# for each digitized line (usually 4 rows)

# generate synthetic signal for debugging
x = np.zeros(500)
x[50:60] = 30
#plt.plot(x)


# 1. diff the signals
xdiff = np.diff(x)
plt.plot(xdiff)

# loop across left, middle and right of signal

#left loop - is there a pulse on the left?
idx = 0
thresh = 25
up = -1
down = -1
while idx < 100:
    if up == -1:
        if xdiff[idx] > thresh:
            up = idx # we expect this to be at the very start of the signal (i.e. idx == 0), but might not be if the digitisation is slightly wrong
    else:  # if an up pulse has been found, then we are either on a false up, a plateau, or about to hit a down pulse 
        if (idx-up) > 25: #min width of pulse we will accept
            if xdiff[idx] < (-thresh):
                down = idx #success!
        else:
            if abs(xdiff[idx]) > 3: # if we actually aren't on a plateau, then this is not a square wave
                up = -1 # reset the squarewave
    idx = idx + 1
                
#right loop - is there a pulse on the right?

#middle loop - is there a pulse in the middle

#logic at the end - is there a loop anywhere?
#if so, what are the start and end indices?
#RETURN: the pixel value of the bottom of the pulse (to get 0mV), start of pulse, end of pulse
                


