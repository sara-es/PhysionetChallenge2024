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

# loop across left, middle and right of signal
def find_pulse(xdiff, start_idx, max_width, tol = 3, min_height = 25, min_width = 25):
    idx = start_idx
    up = -1
    down = -1
    up_y = -1
    
    while idx < (start_idx + max_width):
        if up == -1:
            if xdiff[idx] > min_height:
                up = idx # we expect this to be at the very start of the signal (i.e. idx == 0), but might not be if the digitisation is slightly wrong
                up_y = xdiff[idx]
        else:  # if an up pulse has been found, then we are either on a false up, a plateau, or about to hit a down pulse 
            if (idx-up) > min_width: #min width of pulse we will accept
                if abs(xdiff[idx] + up_y) < tol: # can make this constraint tighter to ensure that the down matches the up
                    down = idx #success!
            else:
                if abs(xdiff[idx]) > tol: # if we actually aren't on a plateau, then this is not a square wave
                    up = -1 # reset the squarewave
                    up_y = -1
        idx = idx + 1
    
    return up, down
    

def find_ref_pulse(signal, max_width = 100):
    xdiff = np.diff(signal)
    # check for square wave on left, middle and right of signal
    l_up, l_down = find_pulse(xdiff,0, max_width) #check left
    m_up, m_down = find_pulse(xdiff,len(xdiff)//2-max_width//2, max_width) #check middle
    r_up, r_down = find_pulse(xdiff,len(xdiff)-max_width, max_width)

    # todo - add sanity checking here - square wave should only exist in one place

    # return the non -1 indices
    up = max([l_up, m_up, r_up])
    down = max([l_down, m_down, r_down])
    baseline = x[up] # TODO - check that this is the right index
    return baseline, up, down

## ----  generate synthetic signal for debugging ----#
x = np.zeros(500)
x[50:80] = 30 # this sohuld return a pulse
baseline, up, down = find_ref_pulse(x)

#RETURN: the pixel value of the bottom of the pulse (to get 0mV), start of pulse, end of pulse
                

## ---- each row will return an up and down index. Check for consistency ----
ups = [up, up, up, up]
downs = [down, down, down, down]

# consensus agreement - if at least two sets of indices match, then accept the pulse
# pulse index might be slightly different (if rotation isn't perfect), so take average if needed
