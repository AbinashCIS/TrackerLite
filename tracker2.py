import cv2
import numpy as np
import dlib
from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import argparse
import logging
import os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s ')
logging.getLogger('apscheduler.executors.default').propagate = False
logging.getLogger('apscheduler.scheduler').propagate = False
logging.addLevelName(
    logging.INFO, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.INFO))

ap = argparse.ArgumentParser()
ap.add_argument("-y",
                "--yolo",
                required=True,
                help="base path to YOLO directory")
ap.add_argument("-i",
                "--input",
                type=str,
                default="",
                help="path to (optional) input video file")
ap.add_argument("-o",
                "--output",
                type=str,
                default="",
                help="path to (optional) output video file")
ap.add_argument("-d",
                "--display",
                type=int,
                default=1,
                help="whether or not output frame should be displayed")
ap.add_argument("-c",
                "--confidence",
                type=float,
                default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t",
                "--threshold",
                type=float,
                default=0.3,
                help="threshold when applying non-maxima suppression")
ap.add_argument("-u",
                "--use-gpu",
                type=bool,
                default=0,
                help="boolean indicating if CUDA GPU should be used")
ap.add_argument("-s",
                "--skip-frames",
                type=int,
                default=10,
                help="# of skip frames between detections")
ap.add_argument("--time",
                type=int,
                default=5,
                help="# of second to indentify as active")
args = vars(ap.parse_args())

with open(os.path.join(args["yolo"], "coco.names"), "rt") as f:
	classes = f.read().rstrip("\n").split("\n")

cfg = os.path.join(args["yolo"], "yolov3-tiny.cfg")
weights = os.path.join(args['yolo'], "yolov3-tiny.weights")

logging.info("Loading Model ...")
net = cv2.dnn.readNetFromDarknet(cfg, weights)

writer = None
W = None
H = None

ct = CentroidTracker(maxDisappeared=40)

trackers = []
trackableObject = {}

prev_sum = 0
prev_roi = np.array([0, 0, 0])
prev_dist = 0
active_obj = {}
totalFrames = 0


def check_active():
	try:
		for id, obj in active_obj.items():
			if "frames_not_active" in obj and obj["frames_not_active"] > 120:
				del active_obj[id]
			elif "frames_active" in obj and obj["frames_active"] >= 1:
				if "time" in obj:
					obj["time"] = obj["time"] + 1
				else:
					obj["time"] = 1
	except RuntimeError:
		pass


scheduler = BackgroundScheduler()
scheduler.start()
scheduler.add_job(func=check_active,
                  args=[],
                  trigger=IntervalTrigger(seconds=1))


def getOutputsNames(net):
	# Get the names of all the layers in the network
	layersNames = net.getLayerNames()
	# Get the names of the output layers, i.e. the layers with unconnected outputs
	return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


video = args["input"]
outputFile = args["output"]
cap = cv2.VideoCapture(video)
vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                             (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while True:
	flag, frame = cap.read()
	rects = []
	if not flag:
		cap.release()
		writer.release()
		break
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		writer = cv2.VideoWriter(outputFile, fourcc, 30.0, (W, H), True)
	if totalFrames % 5 == 0:
		trackers = []
		classIds = []
		confidences = []
		boxes = []
		blob = cv2.dnn.blobFromImage(frame,
		                             1 / 255, (416, 416), [0, 0, 0],
		                             1,
		                             crop=False)
		net.setInput(blob)
		detections = net.forward(getOutputsNames(net))
		frameHeight = frame.shape[0]
		frameWidth = frame.shape[1]
		for out in detections:
			for detection in out:
				scores = detection[5:]
				classId = np.argmax(scores)
				confidence = scores[classId]
				if confidence > args['confidence']:
					center_x = int(detection[0] * frameWidth)
					center_y = int(detection[1] * frameHeight)
					width = int(detection[2] * frameWidth)
					height = int(detection[3] * frameHeight)
					left = int(center_x - width / 2)
					top = int(center_y - height / 2)
					classIds.append(classId)
					confidences.append(float(confidence))
					boxes.append([left, top, width, height])
		indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.2)
		for i in indices:
			i = i[0]
			box = boxes[i]
			left = box[0]
			top = box[1]
			width = box[2]
			height = box[3]
			# Class "person"
			if classIds[i] == 0:
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(left, top, left + width, top + height)
				tracker.start_track(rgb, rect)
				trackers.append(tracker)
				# drawPred(classIds[i], confidences[i], left, top, left + width,
				# top + height)

	else:
		for tracker in trackers:
			tracker.update(rgb)
			pos = tracker.get_position()
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())
			rects.append((startX, startY, endX, endY))
	totalFrames += 1
	objects = ct.update(rects)
	print(active_obj)
	if objects:
		for (objectID, centroid), rect in zip(objects.items(), rects):
			startX, startY, endX, endY = rect
			roi = frame[startY:startY + endY, startX:startX + endX]

			try:
				roi = cv2.resize(roi, (128, 128))
			except:
				continue
			dist = np.linalg.norm(roi - prev_roi)
			dist = (dist - prev_dist) // 1000
			print(dist, objectID)
			if dist in range(-15, 15):
				if objectID in active_obj:
					active_obj[objectID].update({
					    "frames_active":
					    active_obj[objectID]["frames_active"] + 1
					})
				else:
					active_obj[objectID] = {"frames_active": 1}

			prev_roi = roi
			prev_dist = dist
			# print(active_obj)
			to = trackableObject.get(objectID, None)
			if to is None:
				to = TrackableObject(objectID, centroid)
			else:
				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)

				to.centroids.append(centroid)

			trackableObject[objectID] = to
			text = f"ID {objectID}"
			if objectID in active_obj and "time" in active_obj[
			    objectID] and active_obj[objectID]["time"] > args["time"]:
				cv2.rectangle(frame, (startX, startY), (endX, endY),
				              (0, 0, 255), 2)
			else:
				cv2.rectangle(frame, (startX, startY), (endX, endY),
				              (0, 255, 0), 2)
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
						            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
						            2)
	else:
		for id, obj in active_obj.items():
			print(id, obj)
			if "frames_not_active" in obj:
				obj["frames_not_active"] += 1
			else:
				obj["frames_not_active"] = 1
	if writer is not None:
		writer.write(frame)

	if args["display"] == True:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			cap.release()
			writer.release()
			cv2.destroyAllWindows()
			break