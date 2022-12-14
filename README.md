# Competition

## 0. Pre-processing：

### 0-1. Clone Project

````
conda create --name drone python=3.8
git clone https://github.com/AngusBB/Competition.git
cd Competition
````

### 0-2. Clone YOLOv5

````
git clone https://github.com/ultralytics/yolov5.git
````

### 0-3. Clone MMDetection

````
git clone https://github.com/open-mmlab/mmdetection.git
````

### 0-4. Install Reqirements

````
pip install -r requirements.txt
````

### 0-5. Download Dataset

````
mkdir Dataset
````

Please place the official dataset in following structure.

https://drive.google.com/drive/folders/1sCA3ife1VMrB3eI59pjFryXG8k4LKElY?usp=sharing

Please place our custom dataset in following structure.

````
\Competition
   ├── Dataset
   │      ├── Training Dataset_v5
   │      │   ├── train
   │      │   │   ├── img0001.png
   │      │   │   ├── img0001.txt
   │      │   │   ├── img0002.png
   │      │   │   ├── img0002.txt
   │      │   │   └── ...
   │      │   │
   │      │   └── classes.txt
   │      │
   │      ├── public
   │      │   ├── img1001.png
   │      │   ├── img1002.png
   │      │   ├── img1003.png
   │      │   └── ...
   │      │
   │      ├── Private Testing Dataset_v2
   │      │   ├── img1501.png
   │      │   ├── img1502.png
   │      │   ├── img1503.png
   │      │   └── ...
   │      │
   │      ├── SuperResolution_Training_Dataset
   │      │   ├── img0001.png
   │      │   ├── img0002.png
   │      │   ├── img0003.png
   │      │   └── ...
   │      │
   │      ├── SuperResolution_Public_Dataset
   │      │   ├── img1001.png
   │      │   ├── img1002.png
   │      │   ├── img1003.png
   │      │   └── ...
   │      │
   │      ├── SuperResolution_PublicPrivate_Dataset
   │      │   ├── img1501.png
   │      │   ├── img1502.png
   │      │   ├── img1503.png
   │      │   └── ...
   │      │
   │      └── Models
   │          ├── baseline.pt
   │          └── 832x832.pt
   │   
   ├── Code
   ├── main.ipynb
   ├── README.md
   └── requirements.txt
````

### 1. Fulfill the project
Follow the steps in main.ipynb
