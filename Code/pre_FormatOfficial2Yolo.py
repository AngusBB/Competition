import glob
import os
import shutil
import cv2
from tqdm import tqdm


print('1. Renaming Training Dataset_v5 --> Training_Dataset_v5')
OfficialTrainingDatasetRoot = 'Dataset/Training_Dataset_v5/'
os.rename('Dataset/Training Dataset_v5', OfficialTrainingDatasetRoot)


print(f'2. Backing-up Labels to {OfficialTrainingDatasetRoot}labels_origin/...')
os.mkdir(f'{OfficialTrainingDatasetRoot}labels_origin')
LabelsList = sorted(glob.glob(f'{OfficialTrainingDatasetRoot}train/*.txt'))
for labels in tqdm(LabelsList):
    shutil.copy(labels, f'{OfficialTrainingDatasetRoot}labels_origin')


print('3. Formatting Official Labels to yolo Type...')
for labels in tqdm(LabelsList):

    img = cv2.imread(f'{OfficialTrainingDatasetRoot}train/' + labels.split('/')[-1].split('.')[0] + '.png')
    height, width = img.shape[:2]

    f = open(labels, 'r')
    data = []
    data_origin = []
    for line in f.readlines():
        data_origin.append(line)
        lines = line.replace('\n', '').split(',')
        lines[1] = str((float(lines[1]) + float(lines[3]) / 2) / width)
        lines[2] = str((float(lines[2]) + float(lines[4]) / 2) / height)
        lines[3] = str(float(lines[3]) / width)
        lines[4] = str(float(lines[4]) / height)
        data.append(lines)
    f.close()

    f = open(labels, 'w')
    for line in data:
        f.writelines(' '.join(line) + '\n')
    f.close()


print(f'4. Moving Labels to {OfficialTrainingDatasetRoot}labels')
os.mkdir(f'{OfficialTrainingDatasetRoot}labels')
for labels in tqdm(LabelsList):
    os.rename(labels, f'{OfficialTrainingDatasetRoot}labels/' + labels.split('/')[-1])


print(f'5. Renaming {OfficialTrainingDatasetRoot}train --> {OfficialTrainingDatasetRoot}images')
os.rename(f'{OfficialTrainingDatasetRoot}train', f'{OfficialTrainingDatasetRoot}images')
