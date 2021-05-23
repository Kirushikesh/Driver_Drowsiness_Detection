import numpy as np
import cv2
from scipy.spatial import distance as dist
import imutils
import os 

MIN_CONF = 0.3
NMS_THRESH = 0.3
SERIOUS_DISTANCE = 50
ALERT_DISTANCE = 80

def detect_people(frame, net, ln):
	
	(H, W) = frame.shape[:2]
	results = []

  	# construct a blob from the input frame 
	# input dimension is (393, 700, 3) and output dimension is (1, 3, 416, 416) 
	# cv2.dnn.blobFromImage returna 4-dimensional Mat with NCHW dimensions order.
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
	
	#sets the new input for the network
	net.setInput(blob)

	# Runs forward pass to compute output of layer with name outputName here ln.
	# Returns list of 3 outputs each are numpy.ndarrays and the each output shape is
	# 507 x 85 ---> 13 x 13 x 3 x 85
	# 2028 x 85 ---> 26 x 26 x 3 x 85
	# 8112 x 85 ---> 52 x 52 x 3 x 85
	layerOutputs = net.forward(ln)

	boxes = []
	centroids = []
	confidences = []

  	# loop over each of the layer outputs (3)
	for output in layerOutputs:
	
    # loop over each of the detections output shape will be 507 or 2028 or 8112,
	# detection shape will be 85 
		for detection in output:
      		
			# In the detection first 4 are box coordinates and last 80 are class probabilities
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			box = detection[0:4] * np.array([W, H, W, H])

      		# filter detections by ensuring that the object detected was a person and
			# that the minimum confidence is met
			if classID == 0 and confidence > MIN_CONF:
				
				# scale the bounding box coordinates back relative to
				# the size of the image
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top left corner of bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates, centroids, and confidences
				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))

  	# apply non-maxima suppression to suppress weak, overlapping bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)
	
  	# ensure at least one detection exists
	if len(idxs) > 0:

		# idxs.flatten() returns the index of bounding boxes after non max supression.
		for i in idxs.flatten():

			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			
      		# update our results list to consist of the person
			# prediction probability, bounding box coordinates, and the centroid
			r = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(r)

	# return the list of results
	return results

LABELS = open("yolov3_data/coco_classes.txt").read().strip().split("\n")

weightsPath =  "yolov3_data/yolov3.weights"
configPath = "yolov3_data/yolov3.cfg"

#Reads a network model stored in Darknet model file.
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# determine only the *output* layer names that we need from YOLO
# Unlike YOLO and YOLO2, which predict the output at the last layer, 
# YOLOv3 predicts boxes at 3 different scales as illustrated in the below image.
ln = net.getUnconnectedOutLayersNames()

vs = cv2.VideoCapture("demovideo/Test_video_1.mp4")

while True:

	(grabbed, frame) = vs.read()
	if not grabbed:
		break
	
	frame = imutils.resize(frame, width=700)
	results = detect_people(frame, net, ln)
	
	# initialize the set of indexes that violate the minimum social
	# distance
	alert = set()
	serious = set()
	a_lines=list()
	s_lines=list()

	# ensure there are *at least* two people detections (required in
	# order to compute our pairwise distance maps)
	if len(results) >= 2:
        
		# extract all centroids from the results and compute the
        # Euclidean distances between all pairs of the centroids
		centroids = np.array([r[2] for r in results])
        
		#calculate the distance matrix for finding euclidean distance between each centroids
		D = dist.cdist(centroids, centroids, metric="euclidean")
            
        # loop over the upper triangular of the distance matrix
		for i in range(0, D.shape[0]):
            
			for j in range(i + 1, D.shape[1]):
                
				# check to see if the distance between any two
                # centroid pairs is less than the configured number of pixels
				if D[i, j] <= ALERT_DISTANCE:
					if(D[i,j])<= SERIOUS_DISTANCE:
						s_lines.append([centroids[i],centroids[j]])
						serious.add(i)
						serious.add(j)
					else:
						a_lines.append([centroids[i],centroids[j]])
						alert.add(i)
						alert.add(j)

    # loop over the results
	for (i, (prob, bbox, centroid)) in enumerate(results):
		
		# extract the bounding box and centroid coordinates, then
		# initialize the color of the annotation
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0)

		# if the index pair exists within the violation set, then
		# update the color
		if i in alert:
			color = (0,255,255)
		elif i in serious:
			color = (0,0,255)
		
		# draw a bounding box around the person and the
		# centroid coordinates of the person,
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
	
	# draw connecting lines for violating alert people
	for start,end in a_lines:
		cv2.line(frame,start,end,(0,255,255),1)

	# draw connecting lines for violating serious people
	for start,end in s_lines:
		cv2.line(frame,start,end,(0,0,255),1)
    
	# draw the total number of social distancing violations on the
	# output frame
	text = "Social Distancing Violations: "+str(len(serious)+len(alert))
	cv2.putText(frame,text, (10, frame.shape[0] - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
	cv2.imshow("Frame",frame)
    
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):    
		break
    