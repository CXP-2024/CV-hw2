import numpy as np
dx = 1
dy = -1
orientation = np.arctan2(dy, dx)# % (2 * np.pi) 
orientation = np.degrees(orientation)
print(orientation)