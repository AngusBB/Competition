import csv
import glob
import os
import cv2
from tqdm import tqdm

step = 1

if not os.path.isdir('Dataset/Origin_Pubic_Dataset'):
    print(f'{step}. Renaming public --> Origin_Pubic_Dataset')
    os.rename('Dataset/public', 'Dataset/Origin_Pubic_Dataset')
    step += 1


if not os.path.isdir('Dataset/Origin_Private_Dataset'):
    print('2. Renaming Private Testing Dataset_v2 --> Origin_Private_Dataset')
    os.rename('Dataset/Private Testing Dataset_v2', 'Dataset/Origin_Private_Dataset')
    step += 1


DetectRoot = "yolov5/runs/detect/"

DetectList = sorted(glob.glob(DetectRoot + "*"), key=os.path.getmtime)
DetectAnswerList = sorted(glob.glob(DetectList[-1] + "/labels/*"))


print(f'{step}. Start Transforming ' + DetectList[-1].split('/')[-1] + ' to Official Answer Format...')
step += 1

if not os.path.isdir('Output'):
    os.mkdir('Output')

if len(DetectAnswerList) > 500:
    ImageList = sorted(glob.glob('Dataset/Origin_Pubic_Dataset/*')) \
                + sorted(glob.glob('Dataset/Origin_Private_Dataset/*'))
else:
    ImageList = sorted(glob.glob('Dataset/Origin_Pubic_Dataset/*'))

with open('Output/output' + ''.join(char for char in (DetectList[-1].split('/')[-1]) if char not in 'exp')
          + '.csv', 'w', newline='') as csvfile:

    writer = csv.writer(csvfile)

    for file in DetectAnswerList:

        img = cv2.imread(ImageList[int(file.split('/')[-1].split('.')[0][3:]) - 1001])
        height, width = img.shape[:2]

        f = open(file, 'r')
        for line in f.readlines():

            line = line.split(' ')
            temp = [line[0],
                    round((float(line[1]) - float(line[3]) / 2) * width),
                    round((float(line[2]) - float(line[4]) / 2) * height),
                    round(float(line[3]) * width), round(float(line[4]) * height)]

            writer.writerow([file.split('/')[-1].split('.')[0]] + temp)


print(f'{step}. Saving ' + DetectList[-1].split('/')[-1] + ' Official Format Answer in Output/')
