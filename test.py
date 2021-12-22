from  PIL import Image
import  cv2
import numpy
from prediction import *
# image_path="./data/input/WaterMeterNumberRecognition/image/141.jpg"
image_path ="./image/0.jpg"
pred=Prediction()
pred.load_model()
res=pred.predict(image_path)
print(res)
