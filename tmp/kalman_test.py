from filterpy.kalman import KalmanFilter
import numpy as np
x = 20
y = 10
w = 12
h = 12
s = w * h
r = w / float(h)
bbox = np.array([x, y, s, r]).reshape((4, 1))

#%%

kf = KalmanFilter( dim_x= 7 , dim_z = 4 )
kf = KalmanFilter(dim_x=7, dim_z=4) 
kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]]) # state transistion matrix
kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]]) # measurement function

kf.R[2:,2:] *= 10. # measurements noise
kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities # 공분산!! 핵심
kf.P *= 10.
kf.Q[-1,-1] *= 0.01
kf.Q[4:,4:] *= 0.01

kf.x[:4] = bbox
time_since_update = 0

#%%

np.zeros( (1,2))