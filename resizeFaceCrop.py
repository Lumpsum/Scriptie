import time
import flickrapi
from queue import Queue
from threading import Thread
import os
from PIL import Image
import pickle
import shutil
import numpy as np
import dlib
import cv2

class DownloadWorker(Thread):
	def __init__(self, queue):
		Thread.__init__(self)
		self.queue = queue

	def run(self):
		while True:
			# Retrieve from queue and perform tasks
			img = self.queue.get()
			resize(img)
			self.queue.task_done()

def resize(img):
	print('Started: ' +img)

	im = np.array(Image.open('./BodyCropData/'+img))
	img_copy = im.copy()
	gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)          
	faces = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5) 
	left, right, top, bottom = faces[0][0], faces[0][0] + faces[0][2], faces[0][1], faces[0][1] + faces[0][3]
	if top < 0:
		top = 0
	if bottom < 0:
		bottom = 0
	if left < 0:
		left = 0
	if right < 0:
		right = 0

	im = Image.fromarray(im[top:bottom,left:right], 'RGB')
	x, y = im.size
	if x > 224 or y > 224:
	    im.thumbnail((224,224), Image.ANTIALIAS)
	background = Image.new('RGB', (224, 224), 'black')
	background.paste(im, (0, 0))
	background.save('./FaceCropFinal/'+img, 'JPEG')

	print('Finished: ' + img)

if __name__ == "__main__":
	start = time.time()

	def load_obj(name):
		with open(name + '.pkl', 'rb') as f:
			return pickle.load(f)

	images = load_obj('faceNot')
	print(len(images))
	detector = dlib.get_frontal_face_detector()
	haar_face_cascade = cv2.CascadeClassifier('/home/lumpsum/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')

	# Create a queue to communicate with the worker threads
	queue = Queue()
	# Create 8 worker threads
	for x in range(100):
		worker = DownloadWorker(queue)
		# Setting daemon to True will let the main thread exit even though the workers are blocking
		worker.daemon = True
		worker.start()
	# Put the tasks into the queue as a tuple
	for img in images:
		queue.put(img)
	# Causes the main thread to wait for the queue to finish processing all the tasks
	queue.join()

print("done in : ", time.time()-start)