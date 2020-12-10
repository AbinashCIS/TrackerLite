![image info](./tracking.gif)

# Person Tracking

Tracking person in realtime using SSD MobileNet, OpenCV and DLib.

## Requirements

```
APScheduler==3.6.3
astroid==2.4.2
cmake==3.18.4.post1
dlib==19.21.1
imutils==0.5.3
isort==5.6.4
lazy-object-proxy==1.4.3
mccabe==0.6.1
numpy==1.19.4
opencv-python==4.4.0.46
pylint==2.6.0
pytz==2020.4
scipy==1.5.4
six==1.15.0
toml==0.10.2
tzlocal==2.1
wrapt==1.12.1
yapf==0.30.0

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

Run the following commands

```
sudo docker build -t trackerlite .
sudo docker run -it trackerlite
```

## TODO

- Add Yolov4 for Detection
- Fix drawing bounding boxes problem
- ~~Active person in csv file~~
- ~~Red bounding box for active person~~
