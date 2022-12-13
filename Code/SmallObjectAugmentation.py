from ..builder import PIPELINES
import numpy as np
import random


@PIPELINES.register_module()
class SmallObjectAugmentation(object):
    def __init__(self, thresh=64 * 64, prob=0.5, copy_times=3, epochs=30, all_objects=False, one_object=False):

        self.thresh = thresh
        self.prob = prob
        self.copy_times = copy_times
        self.epochs = epochs
        self.all_objects = all_objects
        self.one_object = one_object
        if self.all_objects or self.one_object:
            self.copy_times = 1

    def IsSmallObject(self, h, w):
        if h * w <= self.thresh:
            return True
        else:
            return False

    def ComputeOverlap(self, bbox_a, bbox_b):
        if bbox_a is None: return False
        left_max = max(bbox_a[0], bbox_b[0])
        top_max = max(bbox_a[1], bbox_b[1])
        right_min = min(bbox_a[2], bbox_b[2])
        bottom_min = min(bbox_a[3], bbox_b[3])
        inter = max(0, (right_min - left_max)) * max(0, (bottom_min - top_max))
        if inter != 0:
            return True
        else:
            return False

    def DonotOverlap(self, new_bbox, bboxes):
        for bbox in bboxes:
            if self.ComputeOverlap(new_bbox, bbox):
                return False
        return True

    def CreateCopyLabel(self, h, w, bbox, bboxes):
        bbox = bbox.astype(np.int)
        bbox_h, bbox_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        for epoch in range(self.epochs):
            random_x, random_y = np.random.randint(int(bbox_w / 2), int(w - bbox_w / 2)), \
                                 np.random.randint(int(bbox_h / 2), int(h - bbox_h / 2))
            xmin, ymin = random_x - bbox_w / 2, random_y - bbox_h / 2
            xmax, ymax = xmin + bbox_w, ymin + bbox_h
            if xmin < 0 or xmax > w or ymin < 0 or ymax > h:
                continue
            new_bbox = np.array([xmin, ymin, xmax, ymax]).astype(np.int)

            if self.DonotOverlap(new_bbox, bboxes) is False:
                continue

            return new_bbox
        return None

    def AddPatchInImg(self, bbox, copy_bbox, image):
        copy_bbox = copy_bbox.astype(np.int)
        image[bbox[1]:bbox[3], bbox[0]:bbox[2], :] = image[copy_bbox[1]:copy_bbox[3], copy_bbox[0]:copy_bbox[2], :]
        return image

    def __call__(self, results):
        if self.all_objects and self.one_object:
            return results
        if np.random.rand() > self.prob:
            return results

        img = results['img']
        bboxes = results['gt_bboxes']
        labels = results['gt_labels']

        h, w = img.shape[0], img.shape[1]

        small_object_list = list()
        for idx in range(bboxes.shape[0]):
            bbox = bboxes[idx]
            bbox_h, bbox_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if self.IsSmallObject(bbox_h, bbox_w):
                small_object_list.append(idx)

        l = len(small_object_list)
        # No Small Object
        if l == 0:
            return results

        # Refine the copy_object by the given policy
        # Policy 2:
        copy_object_num = np.random.randint(0, l)
        # Policy 3:
        if self.all_objects:
            copy_object_num = l
        # Policy 1:
        if self.one_object:
            copy_object_num = 1

        random_list = random.sample(range(l), copy_object_num)
        idx_of_small_object = [small_object_list[idx] for idx in random_list]
        select_bboxes = bboxes[idx_of_small_object, :]
        select_labels = labels[idx_of_small_object]

        bboxes = bboxes.tolist()
        labels = labels.tolist()
        for idx in range(copy_object_num):
            bbox = select_bboxes[idx]
            label = select_labels[idx]
            bbox_h, bbox_w = bbox[3] - bbox[1], bbox[2] - bbox[0]

            if self.IsSmallObject(bbox_h, bbox_w) is False: continue

            for i in range(self.copy_times):
                new_bbox = self.CreateCopyLabel(h, w, bbox, bboxes, )
                if new_bbox is not None:
                    img = self.AddPatchInImg(new_bbox, bbox, img)
                    bboxes.append(new_bbox)
                    labels.append(label)

        results['img'] = img
        results['gt_bboxes'] = np.array(bboxes)
        results['gt_labels'] = np.array(labels)
        return results
