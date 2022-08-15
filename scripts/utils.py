import os
import os.path
import cv2
import numpy as np
from PIL import Image

def save_image(folderPath = None, filename = None, image=None):
    if folderPath is None:
        folderPath = "/"
    else:
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)

    im_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    path = os.path.join(folderPath, filename)
    cv2.imwrite(path, im_bgr)




