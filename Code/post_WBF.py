import glob
import os
from ensemble_boxes import *

# ======================================== Input ========================================
candidate_1 = '2'
candidate_2 = '3'
# candidate_3 =

# ======================================== Load Files ========================================
answer_root = "yolov5/runs/detect/"
os.mkdir(f'{answer_root}exp{candidate_1}_{candidate_2}')

candidate_1_root = f'{answer_root}exp{candidate_1}/labels/'
candidate_2_root = f'{answer_root}exp{candidate_2}/labels/'
# candidate_3_root = f'{answer_root}exp{candidate_3}/labels/'

candidate_1_list = sorted(glob.glob(candidate_1_root + "*.txt"))
candidate_2_list = sorted(glob.glob(candidate_2_root + "*.txt"))
# candidate_3_list = sorted(glob.glob(candidate_3_root + "*.txt"))

for num in range(1001, 2001):
    if not os.path.isfile(f'{candidate_1_root}img{num:04d}.txt'):
        f = open(f'{candidate_1_root}img{num:04d}.txt', 'w')
        f.close()
    if not os.path.isfile(f'{candidate_2_root}img{num:04d}.txt'):
        f = open(f'{candidate_2_root}img{num:04d}.txt', 'w')
        f.close()
    # if not os.path.isfile(f'{candidate_3_root}img{num:04d}.txt'):
    #     f = open(f'{candidate_3_root}img{num:04d}.txt', 'w')
    #     f.close()

candidate_1_list = sorted(glob.glob(candidate_1_root + "*.txt"))
candidate_2_list = sorted(glob.glob(candidate_2_root + "*.txt"))
# candidate_3_list = sorted(glob.glob(candidate_3_root + "*.txt"))

if len(candidate_1_list) != 1000:
    print(f'{candidate_1_list}\'s Files missing')
    quit()
if len(candidate_2_list) != 1000:
    print(f'{candidate_2_list}\'s Files missing')
    quit()
# if len(candidate_3_list) != 1000:
#     print(f'{candidate_3_list}\'s Files missing')
#     quit()

# ======================================== WBF ========================================
for i in range(1000):
    # x1, y1, x2, y2.
    bboxes_list = [[], []]
    scores_list = [[], []]
    labels_list = [[], []]
    weights = [1.3, 1]
    iou_thr = 0.5
    skip_box_thr = 0.4
    sigma = 0.1

    f1 = open(candidate_1_list[i], 'r')
    for line in f1.readlines():
        line2float = [float(j) for j in line.replace('\n', '').split(' ')]

        labels_list[0].append(round(line2float[0]))

        bbox = [line2float[1] - line2float[3] / 2,
                line2float[2] - line2float[4] / 2,
                line2float[1] + line2float[3] / 2,
                line2float[2] + line2float[4] / 2]
        # for item in bbox:
        #     if not (item >= 0 and item <= 1):
        #         print(f'{candidate_1_list[i]}: {item}')
        bboxes_list[0].append(bbox)

        scores_list[0].append(line2float[5])
    f1.close()

    f2 = open(candidate_2_list[i], 'r')
    for line in f2.readlines():
        line2float = [float(j) for j in line.replace('\n', '').split(' ')]

        labels_list[1].append(round(line2float[0]))

        bbox = [line2float[1] - line2float[3] / 2,
                line2float[2] - line2float[4] / 2,
                line2float[1] + line2float[3] / 2,
                line2float[2] + line2float[4] / 2]
        # for item in bbox:
        #     if not (item >= 0 and item <= 1):
        #         print(f'{candidate_2_list[i]}: {item}')
        bboxes_list[1].append(bbox)

        scores_list[1].append(line2float[5])
    f2.close()

    # boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
    # boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
    # boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    bboxes, scores, labels = weighted_boxes_fusion(bboxes_list,
                                                   scores_list,
                                                   labels_list,
                                                   weights=weights,
                                                   iou_thr=iou_thr,
                                                   skip_box_thr=skip_box_thr)

    # ======================================== Transform2Yolo ========================================
    data = []
    for j in range(len(labels)):
        data.append(f'{round(labels[j])} {(bboxes[j][0] + bboxes[j][2]) / 2} {(bboxes[j][1] + bboxes[j][3]) / 2} '
                    f'{bboxes[j][2] - bboxes[j][0]} {bboxes[j][3] - bboxes[j][1]} {scores[j]}\n')

    f_ans = open(f'{answer_root}exp{candidate_1}_{candidate_2}/img{(1001 + i):04d}.txt', 'w')
    f_ans.writelines(data)
    f_ans.close()
