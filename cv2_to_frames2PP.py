from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import deque
from sklearn.model_selection import train_test_split
from tflearn.data_utils import to_categorical

from rnn_utils import get_network_wide, get_data
from predict_spatial import predict_on_frames
from rnn_eval import evaluate, load_labels
import argparse
from pdb import set_trace
import numpy as np
import os
import pickle
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tflearn
import scipy.spatial.distance as dist
from tqdm import tqdm
import cv2
import time
import shutil
import pandas as pd

################################# Preprocessing & extraction #################################
def cv2frames(out_folder):
	out_folder = "./"+out_folder+"//"
	
	cap = cv2.VideoCapture(0)
	frate = 12
	i = 0
	sec = 0
	batch_size = 100
	font = cv2.FONT_HERSHEY_SIMPLEX
	time_start = time.time()
	period = 10
	closing_time = time_start + period

	while time.time() < closing_time:
		ret, frame = cap.read()
		if ret == False :
			break

		# cv2.imshow('Capturing the Gesture',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		if i % frate == 0:
			frame = feature_extract(frame)
			sec = sec + 1
			prediction = 'Output'
			cv2.putText(frame, prediction, (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
			cv2.imshow("video", frame)
			cv2.imwrite(out_folder + str(sec)+'.jpg',frame)
			#print(sec)
		i += 1
	cap.release()
	cv2.destroyAllWindows()
	return

def distance_caL(img1, img2):
	dists = []
	for i in range(0, len(img1)):
		dists.append(dist.sqeuclidean(img1[i], img2[i]))
	sum_dists = sum(dists)
	ave_dist = np.sqrt(sum_dists/len(dists))
	return ave_dist

def grouping(out_folder):
	frames = os.listdir(out_folder)
	group_count = 0
	os.mkdir(out_folder+"/"+str(group_count)+"//")
	frame_count = 0
	for frame1, frame2 in zip(frames[:-1],frames[1:]):
		img1 = cv2.imread(out_folder + frame1)
		img2 = cv2.imread(out_folder + frame2)
		img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
		img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

		diff = distance_caL(img1,img2)
		print(diff)
		cv2.imwrite(out_folder+"/"+str(group_count)+"//"+ str(frame_count)+'.jpg',img1)
		if diff > 6000:
			group_count += 1
			frame_count = 0
			os.mkdir(out_folder+"/"+str(group_count)+"//")
		frame_count += 1
	return

def feature_extract(frame):
	# boundaries = [
	#     ([42, 47, 89], [180, 188, 236]),
	#     ([36, 85, 141], [125, 194, 241])
	# ]
	boundaries = [
		([50, 40, 40], [100, 134, 146])
	]
	dsize = (1920,1080)

	lower, upper = boundaries[0]
	lower = np.array(lower, dtype="uint8")
	upper = np.array(upper, dtype="uint8")
	mask1 = cv2.inRange(frame, lower, upper)

	# lower, upper = boundaries[1]
	# lower = np.array(lower, dtype="uint8")
	# upper = np.array(upper, dtype="uint8")
	# mask2 = cv2.inRange(frame, lower, upper)

	# mask = cv2.bitwise_or(mask1, mask2)
	frame = cv2.bitwise_and(frame, frame, mask=mask1)
	print(frame.shape)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	print(frame.shape)
	frame = cv2.resize(frame, dsize)	
	return frame

######################################## init #############################################
if __name__ == "__main__":
    out_folder = "Frames//"
    model_file = "retrained_graph.pb"
    frames_folder = "Frames//"
    input_layer = 'Placeholder'
    output_layer = 'final_result' #'module_apply_default/InceptionV3/Logits/GlobalPool'
    batch_size = int(10)
    train_or_test = "test"
    labels = load_labels("retrained_labels.txt")
    input_data_dump = "predicted-frames-final_result-test.pkl"
    num_frames_per_video = 10
    model_file1 = "non_pool.model"

    ############################ live webcam feed #########################################
    if os.path.exists(out_folder):
        print(1)
        shutil.rmtree(out_folder)
        os.mkdir(out_folder)
    cv2frames(out_folder)
    grouping(out_folder)

    ############################ preprocess frames #########################################
    predictions = predict_on_frames(frames_folder, model_file, input_layer, output_layer, batch_size)

    out_file = 'predicted-frames-%s-%s.pkl' % (output_layer.split("/")[-1], train_or_test)
    print("Dumping predictions to: %s" % (out_file))
    with open(out_file, 'wb') as fout:
        pickle.dump(predictions, fout)
    print("Done.")
    

	# python predict_spatial.py retrained_graph.pb test_frames --output_layer="module_apply_default/InceptionV3/Logits/GlobalPool" --batch=100 --test

    ############################ Evaluate and output #########################################
    evaluate(input_data_dump, num_frames_per_video, batch_size, labels, model_file1)


