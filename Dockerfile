FROM python:3.8.5
WORKDIR /TRACKER_LITE

COPY requirements.txt .

RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y  
COPY tracker.py .
ADD mobilenet_ssd mobilenet_ssd
ADD output output
ADD tracker tracker
ADD videos videos

CMD ["python", "tracker.py", "--prototxt", "mobilenet_ssd/MobileNetSSD_deploy.prototxt", "--model", "mobilenet_ssd/MobileNetSSD_deploy.caffemodel", "--input", "videos/3.mp4", "--output", "output/output_01.avi", "-s", "5"]

# For viewing the video in realtime uncomment the following line comment the previous line
# CMD ["python", "tracker.py", "--prototxt", "mobilenet_ssd/MobileNetSSD_deploy.prototxt", "--model", "mobilenet_ssd/MobileNetSSD_deploy.caffemodel", "--input", "videos/3.mp4", "--output", "output/output_01.avi", "-s", "5", "-v", "True"]