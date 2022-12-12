from sahi.slicing import slice_coco

coco_dict, coco_path = slice_coco(
    coco_annotation_file_path="Dataset/SuperResolution_Training.json",
    image_dir="Dataset/SuperResolution_Training_Dataset/",
    output_coco_annotation_file_name="Dataset/SuperResolution_Training_Sliced.json",
    ignore_negative_samples=False,
    output_dir="Dataset/SuperResolution_Training_Sliced/",
    slice_height=832,
    slice_width=832,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    min_area_ratio=0.1,
    verbose=False
)
