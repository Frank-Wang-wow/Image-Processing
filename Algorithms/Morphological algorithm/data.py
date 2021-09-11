import numpy as np

B1,B5 = np.zeros((3,3),np.uint8),np.zeros((3,3),np.uint8)
B1[:,0] = 1
B1[1,1]=1
B5[1,1]=B5[0,0]=1
B2 = np.rot90(B1,k=1)
B3 = np.rot90(B2,k=1)
B4 = np.rot90(B3,k=1)
B6 = np.rot90(B5,k=1)
B7 = np.rot90(B6,k=1)
B8 = np.rot90(B7,k=1)
Bs = [B1,B2,B3,B4,B4,B6,B7,B8]