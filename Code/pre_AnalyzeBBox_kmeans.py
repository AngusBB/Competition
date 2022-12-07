import time
import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

LabelsList = sorted(glob.glob('Dataset/Origin_Training_Dataset/origin_labels/*'))

Classes = ['car', 'hov', 'person', 'motorcycle']

for Class in Classes:

    print()
    print(f'Reading {Class} BBoxes files...')

    data = np.array([0, 0])
    whether_first_data = True
    for file in LabelsList:

        f = open(file, 'r')

        count = 0
        line3_sum = 0
        line4_sum = 0

        for line in f.readlines():

            if int(line[0]) == Classes.index(Class):
                count += 1

                lines = line.replace('\n', '').split(',')
                line3_sum += float(lines[3])
                line4_sum += float(lines[4])

        if (count != 0) and (whether_first_data == True):
            data = np.array([round(line3_sum / count), round(line4_sum / count)])
            whether_first_data = False
        elif count != 0:
            data = np.vstack((data, np.array([round(line3_sum / count), round(line4_sum / count)])))

        f.close()

    print('There are', len(data), Class + ' mean BBoxes.')

    print('Plotting... ')
    clf = KMeans(n_clusters=3)
    clf.fit(data)
    for i in tqdm(range(0, len(data))):

        if clf.labels_[i] == 0:
            plt.scatter(data[i][0], data[i][1], color='red', s=3)
        elif clf.labels_[i] == 1:
            plt.scatter(data[i][0], data[i][1], color='blue', s=3)
        elif clf.labels_[i] == 2:
            plt.scatter(data[i][0], data[i][1], color='green', s=3)

    plt.autoscale()
    plt.grid()
    plt.savefig(f'3-means_{Class}.png', bbox_inches='tight')
    # plt.show()
