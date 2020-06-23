import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

image = Image.open("segment_0.png")

img = cv.imread('segment_0.png',0)
fld = cv.ximgproc.createFastLineDetector()
lines = fld.detect(img)

result_img = fld.drawSegments(img,lines)

plt.imshow(result_img)
plt.show()
print("hello")

