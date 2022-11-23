# Python code to read image
import cv2
from hocrox.model import Model
from hocrox.layer import Read
from hocrox.layer.preprocessing.transformation import Crop
from hocrox.layer.preprocessing.transformation import Resize
from hocrox.layer.augmentation.flip import RandomFlip
from hocrox.layer.augmentation.transformation import RandomRotate
from hocrox.layer.preprocessing.blur import GaussianBlur
from hocrox.layer.augmentation.color import RandomBrightness
from hocrox.layer.augmentation.transformation import RandomZoom
from hocrox.layer.augmentation.flip import RandomHorizontalFlip
from hocrox.layer.augmentation.flip import RandomVerticalFlip
from hocrox.layer import Save
model = Model()
model.add(Read(path="C:/Users/oatxs/OneDrive/OneDrive - KMITL/Deep_Learning/LangSad/Cropped", name="Read images")) 

# model.add(RandomRotate(start_angle=0, end_angle=270, probability=1.0, number_of_outputs=1, name="Randomly rotates the image"))
# model.add(Save('C:/Users/oatxs/OneDrive/OneDrive - KMITL/Deep_Learning/LangSad/Rotate', 'img',name='LangSad_Rotate'))
# print(model.summary())
# model.transform()

# model.add(GaussianBlur(kernel_size=(9,9), sigma_x=0, sigma_y=0))
# model.add(Save('C:/Users/oatxs/OneDrive/OneDrive - KMITL/Deep_Learning/LangSad/Blur', 'img',name='LangSad_Blur'))
model.add(RandomBrightness(low=0.4, high=1.5, probability=1.0, number_of_outputs=1))
model.add(Save('C:/Users/oatxs/OneDrive/OneDrive - KMITL/Deep_Learning/LangSad/Brightness', 'img',name='LangSad_Brightness'))
print(model.summary())
model.transform()
model.add(RandomHorizontalFlip(probability=1.0, number_of_outputs=1))
model.add(Save('C:/Users/oatxs/OneDrive/OneDrive - KMITL/Deep_Learning/LangSad/FlipLeftRight', 'img',name='LangSad_FlipLeftRight'))
model.add(RandomVerticalFlip(probability=1.0,number_of_outputs=1))
model.add(Save('C:/Users/oatxs/OneDrive/OneDrive - KMITL/Deep_Learning/LangSad/FlipUpDown', 'img',name='LangSad_FlipUpDown'))
model.add(RandomZoom(start=0.2, end=0.7,probability=1.0,number_of_outputs=1))
model.add(Save('C:/Users/oatxs/OneDrive/OneDrive - KMITL/Deep_Learning/LangSad/Scale', 'img',name='LangSad_Scale'))

print(model.summary())
model.transform()