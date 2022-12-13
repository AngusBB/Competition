import glob
from tqdm import tqdm
import cv2
import numpy as np
import os

Classes = ['car', 'hov', 'person', 'motorcycle']

CorrectLabels_root = 'Dataset/CorrectLabels/'
CorrectLabels_list = {
    Classes[0]: sorted(glob.glob(f'{CorrectLabels_root}{Classes[0]}/*.txt')),
    Classes[1]: sorted(glob.glob(f'{CorrectLabels_root}{Classes[1]}/*.txt')),
    Classes[2]: sorted(glob.glob(f'{CorrectLabels_root}{Classes[2]}/*.txt')),
    Classes[3]: sorted(glob.glob(f'{CorrectLabels_root}{Classes[3]}/*.txt'))
}
os.mkdir(f'{CorrectLabels_root}ALL')

SuperResolutionTrainImages_root = 'Dataset/SuperResolution_Training_Dataset/'
SuperResolutionTrainImages_list = sorted(glob.glob(f'{SuperResolutionTrainImages_root}*.png'))

for Class in Classes:
    if len(CorrectLabels_list[Class]) != 1000:
        print(f'{Class}\'s Label Files Incorrect, expect 1000, but {len(CorrectLabels_list[Class])}')
        quit()

for i in tqdm(range(1000)):

    bboxes = []

    WhetherIgnoreFirst = True

    for Class in Classes:
        f = open(CorrectLabels_list[Class][i], 'r')
        for line in f.readlines():
            if line[0] == str(Classes.index(Class)):
                bboxes.append(line)

            # Ignore Area
            elif (line[0] == '4') and (Classes.index(Class) == 3):

                # print(CorrectLabels_list[Class][i])

                if WhetherIgnoreFirst:
                    image = cv2.imread(SuperResolutionTrainImages_list[i])
                    height, width = image.shape[:2]
                    mask = np.ones(shape=[height, width], dtype='uint8')
                    WhetherIgnoreFirst = False

                line2float = [float(j) for j in line.replace('\n', '').split(' ')]
                cv2.rectangle(
                    mask,
                    (round((line2float[1] - line2float[3] / 2) * width),
                     round((line2float[2] - line2float[4] / 2) * height)),
                    (round((line2float[1] + line2float[3] / 2) * width),
                     round((line2float[2] + line2float[4] / 2) * height)),
                    0, -1)

            else:
                print(f'Error Class in {CorrectLabels_list[Class][i]}')

        f.close()

    # White Area
    if not WhetherIgnoreFirst:
        for line in bboxes:
            line2float = [float(j) for j in line.replace('\n', '').split(' ')]
            cv2.rectangle(
                    mask,
                    (round((line2float[1] - line2float[3] / 2) * width),
                     round((line2float[2] - line2float[4] / 2) * height)),
                    (round((line2float[1] + line2float[3] / 2) * width),
                     round((line2float[2] + line2float[4] / 2) * height)),
                    1, -1)
        ImageMask = cv2.bitwise_and(image, image, mask=mask)
        cv2.imwrite(SuperResolutionTrainImages_list[i], ImageMask)

    # print(bboxes)
    f = open(f'Dataset/SuperResolution_Training_Dataset/img{(i+1):04d}.txt', 'w')
    f.writelines(bboxes)
    f.close()
