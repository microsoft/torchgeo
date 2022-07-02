import os

import cv2

image = cv2.imread(os.path.join("images", "M-33-20-D-c-4-2.tif"))
mask = cv2.imread(os.path.join("masks", "M-33-20-D-c-4-2.tif"))

os.makedirs("output")
for i in range(2):
    cv2.imwrite(os.path.join("output", f"M-33-20-D-c-4-2_{i}.jpg"), image)
    cv2.imwrite(os.path.join("output", f"M-33-20-D-c-4-2_{i}_m.png"), mask)
