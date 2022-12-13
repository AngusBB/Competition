import glob
import os
import shutil
import cv2
from tqdm import tqdm
import random


print('1. Renaming Training Dataset_v5 --> Origin_Training_Dataset')
OfficialTrainingDatasetRoot = 'Dataset/Origin_Training_Dataset/'
os.rename('Dataset/Training Dataset_v5', OfficialTrainingDatasetRoot)


print(f'2. Backing-up Labels to {OfficialTrainingDatasetRoot}origin_labels/...')
os.mkdir(f'{OfficialTrainingDatasetRoot}origin_labels')
LabelsList = sorted(glob.glob(f'{OfficialTrainingDatasetRoot}train/*.txt'))
for labels in tqdm(LabelsList):
    shutil.copy(labels, f'{OfficialTrainingDatasetRoot}origin_labels')


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


print(f'4. Moving Labels to {OfficialTrainingDatasetRoot}train/labels...')
os.mkdir(f'{OfficialTrainingDatasetRoot}train/labels')
for labels in tqdm(LabelsList):
    os.rename(labels, f'{OfficialTrainingDatasetRoot}train/labels/' + labels.split('/')[-1])


print(f'5. Moving images to {OfficialTrainingDatasetRoot}train/images...')
os.mkdir(f'{OfficialTrainingDatasetRoot}train/images')
ImagesList = sorted(glob.glob(f'{OfficialTrainingDatasetRoot}train/*.png'))
for images in tqdm(ImagesList):
    os.rename(images, f'{OfficialTrainingDatasetRoot}train/images/' + images.split('/')[-1])


print('6. Start Splitting 10% of Training Data to Val Data...')
os.mkdir(f'{OfficialTrainingDatasetRoot}val')
os.mkdir(f'{OfficialTrainingDatasetRoot}val/images')
os.mkdir(f'{OfficialTrainingDatasetRoot}val/labels')
val_index = random.sample(range(1, 1000), 100)
for index in tqdm(val_index):
    os.rename(f'{OfficialTrainingDatasetRoot}train/images/img{index:04d}.png',
              f'{OfficialTrainingDatasetRoot}val/images/img{index:04d}.png')
    os.rename(f'{OfficialTrainingDatasetRoot}train/labels/img{index:04d}.txt',
              f'{OfficialTrainingDatasetRoot}val/labels/img{index:04d}.txt')
