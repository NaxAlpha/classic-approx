import cv2

import numpy as np


image = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
], dtype=np.uint8)

out1 = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3).astype(np.float32)
out2 = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3).astype(np.float32)
out3 = np.sqrt(out1**2 + out2**2)
out4 = (out3) / (out3.max() + 1e-8)
print(out3)
