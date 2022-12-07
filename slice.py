from sahi.slicing import slice_coco

coco_dict, coco_path = slice_coco(
    coco_annotation_file_path="/home/student/yueh_huangje_2022/Drone/Yolo-to-COCO-format-converter/output/SuperResolution_train.json",
    image_dir="/home/student/yueh_huangje_2022/Drone/Dataset/SuperResolution/",
    output_coco_annotation_file_name="SuperResolution_sliced.json",
    ignore_negative_samples=False,
    output_dir="/home/student/yueh_huangje_2022/Drone/Dataset/SuperResolution_Sliced/",
    slice_height=1200,
    slice_width=1200,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.26,
    min_area_ratio=0.1,
    verbose=False
)

# python pre_FormatYolo2Coco.py --path Dataset/train/Correct/Class_Person --output SuperResolution_Correct_Person_coco.json
# python main.py --path /home/student/yueh_huangje_2022/Drone/Dataset/Train/Correct/Class_Person_All --output SuperResolution_Correct_Person_All_coco.json

# sahi coco yolov5 --image_dir /home/student/yueh_huangje_2022/Drone/Dataset/Train/Correct/Class_Person_Sliced --dataset_json_path /home/student/yueh_huangje_2022/Drone/Dataset/Train/Correct/Class_Person_Sliced/SuperResolution_Correct_Person_sliced_coco.json   --train_split 1
# sahi coco yolov5 --image_dir /home/twsdqna396/drone/Dataset/SuperResolution_Correct_Person_All_sliced --dataset_json_path /home/twsdqna396/drone/Dataset/SuperResolution_Correct_Person_All_sliced/SuperResolution_Correct_Person_All_sliced_coco.json   --train_split 1
