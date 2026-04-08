import numpy as np

arr = np.array([[[[3.0, 1.0], [4.0, 2.0]]]])
print("Before sort:\n", arr)
flat = arr.reshape(arr.shape[0], -1)
flat.sort(axis=1)
print("After sort:\n", arr)
