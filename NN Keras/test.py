import numpy as  np
pred=np.array([[2,0.3],[72,-2]])
pred[0][pred[0]>=0.5]=1
print(pred)