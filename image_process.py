from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import deque
from sklearn.model_selection import train_test_split
from tflearn.data_utils import to_categorical
from rnn_train import train_func
from rnn_utils import get_network_wide, get_data
from predict_spatial import predict_on_frames
from rnn_eval import evaluate, load_labels
import argparse
from retrain import run_retrain
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
import matplotlib.pyplot as plt


def cv2frames(out_folder):
	out_folder = "./"+out_folder+"//"
	if os.path.exists(out_folder):
		shutil.rmtree(out_folder)
		os.mkdir(out_folder)
	else:
		os.mkdir(out_folder)
	cap = cv2.VideoCapture(0)
	frate = 12
	i = 0
	sec = 0
	batch_size = 100
	font = cv2.FONT_HERSHEY_SIMPLEX
	time_start = time.time()
	period = 10
	closing_time = time_start + period
	frames = []
	outputs = []
	while time.time() < closing_time:
		ret, frame = cap.read()
		if ret == False :
			break

		# cv2.imshow('Capturing the Gesture',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		if i == 0:
				backgrd = frame
				cv2.imwrite('/Processed/backgrd.jpg',frame)
				i += 1
				continue
		# if i % frate == 0:
		# 	# frame = feature_extract(frame)
		# 	sec = sec + 1
		# 	prediction = 'Output'
		# 	if i == 0:
		# 		backgrd = frame
		# 		cv2.imwrite('/Processed/backgrd.jpg',frame)
		# 		i += 1
		# 		continue
		out = contour_extract(backgrd, frame)
		cv2.imshow("video", out)
		# cv2.putText(out, prediction, (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
		# cv2.imshow("video", out)
		if i % frate == 0:
			sec = sec + 1
			prediction = 'Output'
			frames.append(frame)
			outputs.append(out)
			cv2.imwrite(out_folder + str(sec)+'.jpg',frame)
			#print(sec)
		i += 1
	cap.release()
	cv2.destroyAllWindows()
	return frames, outputs

def grouping(out_folder):
	frames = os.listdir(out_folder)
	group_count = 0
	os.mkdir(out_folder+"/"+str(group_count)+"//")
	frame_count = 0
	for frame1, frame2 in zip(frames[:-1],frames[1:]):
		img1 = cv2.imread(out_folder +'/' + frame1)
		img2 = cv2.imread(out_folder + '/' + frame2)
		img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
		img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

		diff = distance_caL(img1,img2)
		print(diff)
		cv2.imwrite(out_folder+"/"+str(group_count)+"//"+ str(frame_count)+'.jpg',img1)
		if diff > 2500:
			group_count += 1
			frame_count = 0
			os.mkdir(out_folder+"/"+str(group_count)+"//")
		frame_count += 1
	return

def distance_caL(img1, img2):
	dists = []
	for i in range(0, len(img1)):
		dists.append(dist.sqeuclidean(img1[i], img2[i]))
	sum_dists = sum(dists)
	ave_dist = np.sqrt(sum_dists/len(dists))
	return ave_dist

def feature_extract(frame):
	boundaries = [
		([50, 40, 40], [100, 134, 146])
	]
	dsize = (1920,1080)

	lower, upper = boundaries[0]
	lower = np.array(lower, dtype="uint8")
	upper = np.array(upper, dtype="uint8")
	mask1 = cv2.inRange(frame, lower, upper)

	frame = cv2.bitwise_and(frame, frame, mask=mask1)
	print(frame.shape)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	print(frame.shape)
	frame = cv2.resize(frame, dsize)	
	return frame

def process(in_folder, out_folder):
	in_folder = "./"+in_folder
	out_folder = "./"+out_folder
	backgrd = cv2.imread(out_folder+'ADsub1.jpeg')
	frame = cv2.imread(out_folder+'ADsubges1.jpeg')
	out = contour_extract(backgrd, frame)
	# cv2.imshow("processed",out)
	# cv2.waitKey(2)
	cv2.imwrite(out_folder + 'processed.jpg',out)
	cv2.destroyAllWindows()
	return

def contour_extract(backgrd, frame):
	backgrd = cv2.cvtColor(backgrd, cv2.COLOR_BGR2GRAY)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	if backgrd.shape != frame.shape:
		dsize = backgrd.shape[::-1]
		frame = cv2.resize(frame, dsize)
	frame = abs(backgrd-frame)
	out_mask = np.zeros_like(frame)
	out=frame.copy()
	################## Lower end ########################
	median = np.median(frame[frame > 0])
	median = max(50,median)
	median = min(80,median)
	print("Median :", median)
	mask = frame.copy()
	mask[mask > median+20] = 0
	mask[mask < median-20] = 0
	mask[mask > 0] = 255
	contours, _ = cv2.findContours(mask,2,1)
	contours = sorted(contours, key=cv2.contourArea)
	cv2.drawContours(out_mask, contours[-3:], -1, 255, cv2.FILLED, 1)
	out[out_mask == 0] = 0
	################## Higher end ########################
	median = np.median(frame[frame > 0])
	median = max(180,median)
	median = min(210,median)
	print("Median :", median)
	mask = frame.copy()
	mask[mask > median+20] = 0
	mask[mask < median-20] = 0
	mask[mask > 0] = 255
	contours, _ = cv2.findContours(mask,2,1)
	contours = sorted(contours, key=cv2.contourArea)
	cv2.drawContours(out_mask, contours[-3:], -1, 255, cv2.FILLED, 1)
	out[out_mask == 0] = 0
	return out

def vid2frame(vidIn, vframes):
	videos = os.listdir(vidIn)
	
	videos = [video for video in videos]
	os.chdir(vidIn)
	for video in tqdm(videos, unit='videos', ascii=True):
		# name = os.path.abspath(video)
		cap = cv2.VideoCapture(video)  # capturing input video
		count = 0
		i = 0
		frate = 10
		# assumption only first 10 frames are important
		while count < 10:
			ret, frame = cap.read()
			print(ret)
			if (ret is False) or (frame is None) :
				count -= 1
				break
			
			# if not os.path.exists(vframes):
				#frame = cv2.resize(frame,(1920,1080))
			if i % frate == 0:
				framename = '.' + vframes +str(video)+ str(count) + '.jpg'
				cv2.imwrite(framename, frame)
				count += 1
			i += 1
	return

def vidgroups(gesture_folder, target_folder):
	rootPath = os.getcwd()
	template_folder = os.path.join(rootPath, 'Processed')
	majorData = os.path.abspath(target_folder)
	if not os.path.exists(majorData):
		os.makedirs(majorData)
	gesture_folder = os.path.abspath(gesture_folder)
	os.chdir(gesture_folder)
	gestures = os.listdir(os.getcwd())
	print("Source Directory containing gestures: %s" % (gesture_folder))
	print("Destination Directory containing frames: %s\n" % (majorData))
	for gesture in tqdm(gestures, unit='actions', ascii=True):
		gesture_path = os.path.join(gesture_folder, gesture)
		os.chdir(gesture_path)
		gesture_frames_path = os.path.join(majorData, gesture)
		if not os.path.exists(gesture_frames_path):
			os.makedirs(gesture_frames_path)
		videos = os.listdir(os.getcwd())
		videos = [video for video in videos if(os.path.isfile(video))]
		for video in tqdm(videos, unit='videos', ascii=True):
			print(video)
			name = os.path.abspath(video)
			cap = cv2.VideoCapture(name)  # capturing input video
			os.chdir(gesture_frames_path)
			count = 0
			i = 0
			frate = 10
			# assumption only first 10 frames are important
			while count < 10:
				ret, frame = cap.read()
				if (ret is False) or (frame is None) :
					count -= 1
					break
				if i % frate == 0:
					if video[:2] == 'AD':
						backgrd = cv2.imread(template_folder+'/ADsub.jpg')
					elif video[:2] == 'SB':
						backgrd = cv2.imread(template_folder+'/SBsub.jpg')
					else:
						backgrd = cv2.imread(template_folder+'/sub.jpg')
					frame = contour_extract(backgrd, frame)
					framename = os.path.splitext(video)[0]
					framename = framename + "_frame_" + str(count) + ".jpeg"
					# framename = '.' + vframes +str(video)+ str(count) + '.jpg'
					cv2.imwrite(framename, frame)
					count += 1
				i += 1
			os.chdir(gesture_path)
			cap.release()
			cv2.destroyAllWindows()
	os.chdir(rootPath)

######################################## main #############################################
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', action='store_true', help='passed if frames_folder belongs to test_data')
	args = parser.parse_args()
	if args.train:
		train_or_test = "train"
	else:
		train_or_test = "test"
	out_folder = "test_frames"
	out_process = "/Processed//"
	vidIn = './Train videos//'
	vframes = './vidFrames//'
	gesture_folder = 'train_videos'
	target_folder = 'train_frames'
	model_file = "retrained_graph.pb"
	input_layer = 'Placeholder'
	output_layer = 'module_apply_default/InceptionV3/Logits/GlobalPool' #'final_result' 
	batch_size = int(10)
	input_data_dump = "predicted-frames-{}-{}.pkl".format(output_layer.split("/")[-1], train_or_test)
	labels = load_labels("retrained_labels.txt")
	num_frames_per_video = 10
	model_file1 = "pool.model" #"non_pool.model"
	# cv2frames(out_folder)
	# vid2frame(vidIn, vframes)
	# process(out_folder,out_process)
	if train_or_test == 'train':
		print('------------ Training --------------')
		vidgroups(gesture_folder, target_folder)
		# python3 retrain.py --bottleneck_dir=bottlenecks --summaries_dir=training_summaries/long --output_graph=retrained_graph.pb --output_labels=retrained_labels.txt --image_dir=train_frames
		global_vars = {}
		local_vars = {	
						"bottleneck_dir":"bottlenecks",
						"summaries_dir":"training_summaries/long",
						"output_graph":"retrained_graph.pb",
						"output_labels":"retrained_labels.txt",
						"image_dir":"train_frames"
					}
		# with open("retrain.py") as f:
		# 	code = compile(f.read(), "retrain.py", 'exec')
		# 	exec(code, global_vars, local_vars)
		run_retrain()
		predictions = predict_on_frames(target_folder, model_file, input_layer, output_layer, batch_size)
	else:
		print('------------ Testing --------------')
		frames, out_frames = cv2frames(out_folder)
		grouping(out_folder)
		predictions = predict_on_frames(out_folder, model_file, input_layer, output_layer, batch_size)
	out_file = 'predicted-frames-%s-%s.pkl' % (output_layer.split("/")[-1], train_or_test)
	print("Dumping predictions to: %s" % (out_file))
	with open(out_file, 'wb') as fout:
		pickle.dump(predictions, fout)
	print("Done.")
	if train_or_test == 'train':
		train_func(input_data_dump, num_frames_per_video, batch_size, labels, model_file1)
	if train_or_test == 'test':
		output = evaluate(input_data_dump, num_frames_per_video, batch_size, labels, model_file1)
		print(labels)
		rev_labels = dict(zip(list(labels.values()), list(labels.keys())))			
		font = cv2.FONT_HERSHEY_SIMPLEX
		# set_trace()
		for frame in frames:
			# print(frame.shape)
			cv2.putText(frame, rev_labels[output[0]], (50, 50), font, 1, (255, 0, 255), 2, cv2.LINE_4)
			cv2.imshow("video", frame)
			cv2.waitKey(500)
	cv2.destroyAllWindows()