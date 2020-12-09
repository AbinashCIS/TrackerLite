# Person Tracking

Tracking person in realtime using SSD MobileNet, OpenCV and DLib.

## Requirements

```
cmake==3.18.4.post1
dlib==19.21.1
imutils==0.5.3
numpy==1.19.4
opencv-python==4.4.0.46
scipy==1.5.4
```

## Installation

```
pyhton3 -m venv env
source env/bin/activate
pip install -r requirements.txt

```

## Run

```
python tracker.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/1.mp4 --output output/output_01.avi  -s 5

```

## Docker

Comming soon
