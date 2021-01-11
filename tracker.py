from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse

import imutils
import time
import cv2
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import dlib
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s ')
logging.getLogger('apscheduler.executors.default').propagate = False
logging.getLogger('apscheduler.scheduler').propagate = False
logging.addLevelName(
    logging.INFO, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.INFO))

ap = argparse.ArgumentParser()
ap.add_argument("-p",
                "--prototxt",
                required=True,
                help="path to Caffe 'deploy prototxt file")
ap.add_argument("-m",
                "--model",
                required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-i",
                "--input",
                type=str,
                help="path to optional input video file")
ap.add_argument("-o",
                "--output",
                type=str,
                help="path to optional output video file")
ap.add_argument("-c",
                "--confidence",
                type=float,
                default=0.4,
                help="minimum probability to filter weak detections")
ap.add_argument("-s",
                "--skip-frames",
                type=int,
                default=30,
                help="# of frame skips between detections")
ap.add_argument("-t",
                "--time",
                type=int,
                default=5,
                help="number of seconds to count as a active person")
ap.add_argument("-v",
                "--view",
                type=bool,
                default=False,
                help="view realtime image processing")
args = vars(ap.parse_args())

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

logging.info("Loading model ... ")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

if not args.get("input", False):
	logging.info("Starting video stream ... ")
	vs = VideoStream(src=0).start()
	time.sleep(2)
else:
	logging.info("Opening video file ... ")
	vs = cv2.VideoCapture(args["input"])

logging.info("Processing video ... ")

with open("output/result.csv", "w") as f:
	f.write("Time,People Count Changed,TotalCount,ActivePerson,\n")

writer = None
W = None
H = None

ct = CentroidTracker(maxDisappeared=60, maxDistance=30)

trackers = []
trackableObject = {}

totalFrames = 0
totalDown = 0
totalUp = 0

fps = FPS().start()

prev_sum = 0
prev_roi = np.array([0, 0, 0])
prev_dist = 0
active_obj = {}
people_count = 0
total_count = 0


def check_active():
	for obj in active_obj.values():
		print(active_obj)
		if "time" in obj and obj["frames"] > 15:
			obj["time"] = obj["time"] + 1
		elif obj["frames"] > 20:
			obj["time"] = 1


def write_csv():
	with open("output/result.csv", "a") as f:
		f.write(
		    f"{datetime.now()},{len(active_obj) if len(active_obj) > people_count else 0},{len(active_obj) if len(active_obj) > total_count else total_count}\n"
		)


scheduler = BackgroundScheduler()
scheduler.start()

scheduler.add_job(func=check_active,
                  args=[],
                  trigger=IntervalTrigger(seconds=1))
scheduler.add_job(func=write_csv, args=[], trigger=IntervalTrigger(seconds=2))

while True:
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame

	status = "Waiting"
	rects = []
	if args["input"] is not None and frame is None:
		break
	frame = imutils.resize(frame, width=500)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		writer = cv2.VideoWriter(args["output"], fourcc, 30.0, (W, H), True)

	if totalFrames % args["skip_frames"] == 0:
		status = "Detecting"
		trackers = []

		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		net.setInput(blob)
		detections = net.forward()

		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > args["confidence"]:
				idx = int(detections[0, 0, i, 1])
				if CLASSES[idx] != "person":
					continue
				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(rgb, rect)
				trackers.append(tracker)
	else:
		for tracker in trackers:
			status = "Tracking"
			tracker.update(rgb)
			pos = tracker.get_position()
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			rects.append((startX, startY, endX, endY))

	objects = ct.update(rects)
	print(objects, rects)

	for objID, _ in active_obj.items():
		if objID not in objects:
			# del active_obj[objID]
			active_obj[objID] = {"frames": 1, "last_frame": 0, "no_change": 0}
		# if obj not in objts:
		# 	del active_obj[obj]

	for (objectID, centroid), rect in zip(objects.items(), rects):
		startX, startY, endX, endY = rect
		roi = frame[startY:startY + endY, startX:startX + endX]

		try:
			roi = cv2.resize(roi, (128, 128))
		except:
			continue
		dist = np.linalg.norm(roi - prev_roi)
		dist = (dist - prev_dist) // 1000
		# print(dist, active_obj)
		if dist in range(0, 30):
			if objectID in active_obj:
				active_obj[objectID].update({
				    "last_frame":
				    active_obj[objectID]["frames"],
				    "frames":
				    active_obj[objectID]["frames"] + 1
				})
			else:
				active_obj[objectID] = {
				    "frames": 1,
				    "last_frame": 0,
				    "no_change": 0
				}
			# active_obj2 = active_obj
			# for objectID in active_obj2:
			# 	last = active_obj[objectID]["last_frame"]
			# 	if (last + 1) == active_obj[objectID]["frames"]:
			# 		active_obj.update(
			# 		    {"no_change": active_obj[objectID]["no_change"] + 1})
			# 	if active_obj[objectID]["no_change"] > 5:
			# 		del active_obj[objectID]

		prev_roi = roi
		prev_dist = dist

		to = trackableObject.get(objectID, None)

		if to is None:
			to = TrackableObject(objectID, centroid)
		else:
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)

			to.centroids.append(centroid)

			if not to.counted:
				if direction < 0 and centroid[1] < H // 2:
					totalUp += 1
					to.counted = True
				elif direction > 0 and centroid[1] > H // 2:
					totalDown += 1
					to.counted = True
		trackableObject[objectID] = to
		text = f"ID {objectID}"

		if objectID in active_obj and "time" in active_obj[
		    objectID] and active_obj[objectID]["time"] > 5:
			cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255),
			              2)
		else:
			cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0),
			              2)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
		for obj in active_obj.values():
			if "time" in obj:
				cv2.putText(frame, str(obj["time"]),
				            (centroid[0] + 30, centroid[1] - 110),
				            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
				if obj["time"] > args["time"]:
					cv2.putText(frame, "Active",
					            (centroid[0] - 28, centroid[1] - 30),
					            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

	if writer is not None:
		writer.write(frame)

	if args["view"] == True:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
	totalFrames += 1
	fps.update()

fps.stop()
logging.info(f"Elasped time: {fps.elapsed()}")
logging.info(f"Approx FPS: {fps.fps()}")

if writer is not None:
	writer.release()

if not args.get("input", False):
	vs.stop()
else:
	vs.release()
cv2.destroyAllWindows()