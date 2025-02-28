import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import datetime

# Given 3 points, find the parabola that passes through them
#Method 1: without lagrange interpolation
def parabola_1(x1, y1, x2, y2, x3, y3):
    time1 = datetime.datetime.now()
    A = np.array([[x1**2, x1, 1], [x2**2, x2, 1], [x3**2, x3, 1]])
    b = np.array([y1, y2, y3])
    s = np.linalg.solve(A, b)
    time2 = datetime.datetime.now()
    print(time2-time1)
    print(f'y = {s[0]}x^2 + {s[1]}x + {s[2]}')

#Method 2: with lagrange interpolation
def parabola_2(x1, y1, x2, y2, x3, y3):
    time1 = datetime.datetime.now()
    A = np.array([[(x1-x2)*(x1-x3),0,0],[0,(x2-x1)*(x2-x3),0],[0,0,(x3-x2)*(x3-x1)]])
    b = np.array([y1, y2, y3])
    s = [y1/(A[0][0]), y2/(A[1][1]), y3/(A[2][2])]
    time2 = datetime.datetime.now()
    print(time2-time1)
    print(f'y = {s[0]}(x1 - x2) (x1 - x3) + {s[1]}(x2 - x1) (x2 - x3) + {s[2]}(x3 - x2) (x3 - x1)') 
    

parabola_1(1, 2, 2, 3, 3, 4)
parabola_2(1, 2, 2, 3, 3, 4)
