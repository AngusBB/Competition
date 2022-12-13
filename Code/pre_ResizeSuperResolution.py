import glob
from tqdm import tqdm
import cv2

sr_dataset_root = "Dataset/"

ImagesList = sorted(glob.glob(sr_dataset_root + "SuperResolution_Training_Dataset/*.png")) \
             + sorted(glob.glob(sr_dataset_root + "SuperResolution_Public_Dataset/*.png")) \
             + sorted(glob.glob(sr_dataset_root + "SuperResolution_PublicPrivate_Dataset/*.png"))

print('Super-resolving all images to 3840x2160...')
for image in tqdm(ImagesList):

    img = cv2.imread(image)
    if img.shape[0] == 1440:
        print('  Resizing ', image)
        img_resized = cv2.resize(img, (3840, 2160), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(image, img_resized)
