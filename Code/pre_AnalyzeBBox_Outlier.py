import os.path
from math import floor
import seaborn as sns
import matplotlib.pyplot as plt
import glob


LabelsList = sorted(glob.glob('Dataset/Origin_Training_Dataset/origin_labels/*.txt'))

Classes = ['car', 'hov', 'person', 'motorcycle']

for file in LabelsList:

    if os.path.getsize(file) == 0:
        continue

    for Class in Classes:

        data = []

        f = open(file, 'r')
        for line in f.readlines():
            if int(line[0]) == Classes.index(Class):
                line = line.replace('\n', '').split(',')
                area = float(line[3]) * float(line[4])
                data.append(area)
        f.close()

        data_length = len(data)

        if data_length < 4:
            continue

        # sns.boxplot(data=data)
        # plt.show()

        data.sort()

        IQR = data[floor(data_length*3/4)] - data[floor(data_length/4)]
        limit_up = data[floor(data_length*3/4)] + IQR * 4
        limit_down = data[floor(data_length/4)] - IQR * 4

        # print(IQR)
        if max(data) > limit_up and min(data) < limit_down:
            print(f'There may be incorrect labels {Class:>10} in', file.split('/')[-1],
                  f'with max area {int(max(data)):5d} and min area {int(min(data)):5d}')
        elif max(data) > limit_up:
            print(f'There may be incorrect labels {Class:>10} in', file.split('/')[-1],
                  f'with max area {int(max(data)):5d}')
        elif min(data) < limit_down:
            print(f'There may be incorrect labels {Class:>10} in', file.split('/')[-1],
                  f'with                    min area {int(min(data)):5d}')
