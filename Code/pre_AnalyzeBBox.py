import glob
import matplotlib.pyplot as plt
import numpy as np

LabelsList = sorted(glob.glob('Dataset/Origin_Training_Dataset/labels_origin/*'))

data = []

for file in LabelsList:

    f = open(file, 'r')
    for line in f.readlines():
        lines = line.replace('\n', '').split(',')
        data.append(int(lines[3]) * int(lines[4]))
    f.close()


def plotting(bbox):
    print("min =", min(bbox), "; max =", max(bbox), "; mean =", np.mean(bbox), "; length =", len(bbox))

    x = {'0~9': sum(i <= 9 for i in bbox)}
    for i in range(5, 36, 2):
        x[f'{sum(j for j in range(3, i-1, 2))}^2~{sum(j for j in range(3, i+1, 2))}^2'] \
            = sum(sum(j for j in range(3, i-1, 2))**2 < k <= sum(j for j in range(3, i+1, 2))**2 for k in bbox)

    plt.figure(figsize=(15, 15))
    plt.title('BBox')
    plt.xticks(rotation=65)
    rects = plt.bar(np.arange(len(list(x.keys()))), list(x.values()), tick_label=list(x.keys()),
                    color=['r', 'r', 'r', 'r', 'r', 'r',
                           'g', 'g', 'g', 'g', 'g',
                           'b', 'b', 'b', 'b', 'b', 'b'])
    plt.xlabel('Areas Ranges')
    plt.ylabel('BBoxes Amount')

    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha='center', va='bottom')


plotting(data)
plt.savefig('Analyze-Mistaken_BBox.png')
