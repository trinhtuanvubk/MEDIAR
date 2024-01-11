import numpy as np
import cv2


image = cv2.imread("./hihi.jpg")
np.save("hihi.npy",image)