import cv2
import numpy as np 
#import os
#import sys
import dlib
#from skimage import io 
#import glob
import imageio

TRAINING = False
DETECTING = True
# set options for SVM training model

if TRAINING == True :
	options = dlib.simple_object_detector_training_options()

	options.add_left_right_image_flips = True
	options.C = 5
	options.num_threads = 4
	options.be_verbose = True

	dlib.train_simple_object_detector("dataset.xml","detector.svm", options)

	print("Training accuracy: {}".format(dlib.test_simple_object_detector("dataset.xml","detector.svm")))

if DETECTING == True :

	detector = dlib.simple_object_detector("detector.svm")

	#win_det = dlib.image_window()
	#win_det.set_image(detector)

	#win = dlib.image_window()
	'''
	for f in glob.glob(os.path.join("/home/thangnx/HD/code/","*.jpg")):
		img = io.imread(f)
		dets = detector(img)

		for k, d in enumerate(dets):
			print("detection {}: LEFT: {} TOP: {} RIGHT: {} BOTTOM:{} ".format(k, d.left(), d.top(), d.right(),d.bottom()))

		win.clear_overlay()
		win.set_image(img)
		win.add_overlay(dets)
		#dlib.hit_enter_to_continue()
	'''

	cam = cv2.VideoCapture("video2.mp4")
	#cam = cv2.VideoCapture(0)
	#fourcc = cv2.VideoWriter_fourcc(*'XVID')
	#ret,frame = cam.read()
	#size = (frame.shape[0],frame.shape[1])
	#fourcc = cv2.cv.CV_FOURCC(*'XVID')
	#out = cv2.VideoWriter("output.avi",fourcc,20.0,(640,480))

	writer = imageio.get_writer("out2.mp4")


	while(cam.isOpened()):
		ret,frame = cam.read()
		dets = detector(frame)
		#frame = cv2.flip(frame,0)


		for k, d in enumerate(dets):
		#	print("detection {}: LEFT: {} TOP: {} RIGHT: {} BOTTOM:{} ".format(k, d.left(), d.top(), d.right(),d.bottom()))
			cv2.rectangle(frame, (d.left(), d.top()),(d.right(),d.bottom()), (255,0,255),2)
		#cv2.imshow("result",frame)
		#cv2.waitKey(20)
		##win.clear_overlay()
		##win.set_image(frame)
		#cv2.imshow("result",frame)
		#cv2.waitKey(20)
		##win.add_overlay(dets)

		cv2.imshow("result",frame)

		#frame = cv2.flip(frame,180)
		#out.write(frame)
		if ret == True:
			#frame = cv2.flip(frame,0)
			#out.write(frame)
			writer.append_data(frame)

			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
		else:
			break
		#cv2.waitKey(1)

	cam.release()
	#out.release()
	writer.close()
	cv2.destroyAllWindows()



