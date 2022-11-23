import imgaug.augmenters as iaa
import cv2 as cv
import glob

img_counter = 0

# Load DataSet
images_path = glob.glob(
    "C:/Users/oatxs/OneDrive/OneDrive - KMITL/Deep_Learning/LongGong/Cropped/*.JPG")
total_images = len(images_path)

# Image Augmentation
augmentation = iaa.Sequential([
    # iaa.Sometimes(0.7,
    # iaa.Fliplr(1)
    # ),

    # iaa.Sometimes(0.7,
    # iaa.Flipud(1)
    # ),

    # iaa.Sometimes(0.7,
    # iaa.Affine(scale=(1.2, 1.5)),
    # ),

    # iaa.Sometimes(0.7,
    #     iaa.GaussianBlur((9.0, 19.0)),
    # ),

    # iaa.Sometimes(0.7,
        iaa.Multiply((0.4, 1.5))
    # ),

    # iaa.LinearContrast((0.6, 1.4))

    # iaa.Sometimes(0.8,
    #     iaa.Sometimes(0.33,
    #     iaa.Affine(rotate=(-90, -90)),
    #     iaa.Sometimes(0.5,
    #         iaa.Affine(rotate=(90, 90)),
    #         iaa.Affine(rotate=(180, 180)),
    #     )
    # )
    # )
    
])

print('Total : ' + str(total_images))
for image_path in images_path:
    img_counter += 1
    print('Processing ' + str(round(img_counter/total_images*100, 2)) + " % " + 'Remaining : '+str(img_counter)+'/'+str(total_images))
    images = []
    img = cv.imread(image_path)
    images.append(img)#

    augmented_image = augmentation(images=images)

    img_name = "C:/Users/oatxs/OneDrive/OneDrive - KMITL/Deep_Learning/LongGong/Brightness/LongGong_Brightness_{}.JPG".format(img_counter)
    cv.imwrite(img_name, augmented_image[0])
    

print("All Done !!!")
