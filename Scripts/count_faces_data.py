import urllib.request
import time
from multiprocessing import Pool
import flickrapi
from queue import Queue
from threading import Thread
import os
import cv2
import dlib
import shutil

img = cv2.imread('./Flickr/aaron carter/aaron carter0.jpg')
detector = dlib.get_frontal_face_detector()
dets = detector(img, 1)
len(dets)

class DownloadWorker(Thread):
   def __init__(self, queue):
       Thread.__init__(self)
       self.queue = queue

   def run(self):
       while True:
           # Retrieve from queue and perform tasks
           text = self.queue.get()
           filter_images(text)
           self.queue.task_done()

def filter_images(text):
	for x in os.listdir('./Flickr/' + text):
		img = cv2.imread('./Flickr/' + text + '/' + x)
		detector = dlib.get_frontal_face_detector()
		dets = detector(img, 1)
		if len(dets) == 1:
			shutil.copy2('./Flickr/' + text + '/' + x, './FilteredFlickr/' + text + '/' + x)

	print('Finished: ' + text)

if __name__ == "__main__":
	start = time.time()

	celebs = [x for x in os.listdir('./Flickr')]

	# Create a queue to communicate with the worker threads
	queue = Queue()
	# Create 8 worker threads
	for x in range(100):
		worker = DownloadWorker(queue)
		# Setting daemon to True will let the main thread exit even though the workers are blocking
		worker.daemon = True
		worker.start()
	# Put the tasks into the queue as a tuple
	for text in celebs:
		queue.put(text)
	# Causes the main thread to wait for the queue to finish processing all the tasks
	queue.join()

print("done in : ", time.time()-start)