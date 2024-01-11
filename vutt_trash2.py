import numpy as np
import cv2


image = cv2.imread("./hihi.jpg")
print(image.shape)
np.save("hihi.npy",image)
a = np.load("hihi.npy")
print(a.shape)