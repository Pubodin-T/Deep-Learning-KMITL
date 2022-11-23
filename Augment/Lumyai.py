import imgaug.augmenters as iaa
import cv2 as cv
import glob

img_counter = 0

# Load DataSet
images_path = glob.glob(
    "C:/Users/oatxs/OneDrive/OneDrive - KMITL/Deep_Learning/LumYai/Cropped/*.JPG")
total_images = len(images_path)

# Image Augmentation
augmentation = iaa.Sequential([
    iaa.Fliplr(1)
])

print("Hello LumYai")
print('Total : ' + str(total_images))
for image_path in images_path:
    img_counter += 1
    print('Processing ' + str(round(img_counter/total_images*100, 2)) + " % " + 'Remaining : '+str(img_counter)+'/'+str(total_images))
    images = []
    img = cv.imread(image_path)
    images.append(img)

    augmented_image = augmentation(images=images)

    img_name = "C:/Users/oatxs/OneDrive/OneDrive - KMITL/Deep_Learning/LumYai/FlipLeftRight/LumYai_FlipLeftRight_{}.JPG".format(img_counter)
    cv.imwrite(img_name, augmented_image[0])


print("All Done !!!")
