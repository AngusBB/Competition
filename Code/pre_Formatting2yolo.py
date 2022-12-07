import glob
import os
import cv2
from tqdm import tqdm


TrainingDatasetOriginRoot = 'Dataset/Training Dataset_v5/train/'
LabelsList = sorted(glob.glob(f'{TrainingDatasetOriginRoot}*.txt'))
# print(LabelsList)
os.mkdir('Dataset/Training Dataset_v5/yoloLabels')

print('Formatting Official Labels to yolo Type...')
for labels in tqdm(LabelsList):

    img = cv2.imread(TrainingDatasetOriginRoot + labels.split('/')[-1].split('.')[0] + ".png")
    height, width = img.shape[:2]

    f = open(labels, 'r')
    text = []
    for line in f.readlines():
        lines = line.replace('\n', '').split(',')
        lines[1] = str((float(lines[1]) + float(lines[3]) / 2) / width)
        lines[2] = str((float(lines[2]) + float(lines[4]) / 2) / height)
        lines[3] = str(float(lines[3]) / width)
        lines[4] = str(float(lines[4]) / height)
        text.append(lines)
    f.close()

    f = open('Dataset/Training Dataset_v5/yoloLabels/' + labels.split('/')[-1], 'w')
    for line in text:
        f.writelines(' '.join(line) + '\n')
    f.close()
