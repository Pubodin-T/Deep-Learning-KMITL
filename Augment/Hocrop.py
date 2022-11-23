# Python code to read image
import cv2
from hocrox.model import Model
from hocrox.layer import Read
from hocrox.layer.preprocessing.transformation import Crop
from hocrox.layer.preprocessing.transformation import Resize
from hocrox.layer.augmentation.flip import RandomFlip
from hocrox.layer.augmentation.transformation import RandomRotate
from hocrox.layer import Save
model = Model()
model.add(Read(path="C:/Users/oatxs/OneDrive/OneDrive - KMITL/Deep_Learning/LangSad/Default", name="Read images")) 
model.add(Crop(x=0, y=1000, w=4000, h=4000))
model.add(Save('C:/Users/oatxs/OneDrive/OneDrive - KMITL/Deep_Learning/LangSad/Cropped', 'img','LangSad_Cropped'))
model.transform()
model = Model()
model.add(Read(path="C:/Users/oatxs/OneDrive/OneDrive - KMITL/Deep_Learning/LongGong/Default", name="Read images")) 
model.add(Crop(x=0, y=1000, w=4000, h=4000))
model.add(Save('C:/Users/oatxs/OneDrive/OneDrive - KMITL/Deep_Learning/LongGong/Cropped', 'img','LongGong_Cropped'))
print(model.summary())
model.transform()