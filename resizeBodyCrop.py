import time
import flickrapi
from queue import Queue
from threading import Thread
import os
from PIL import Image
import pickle
import shutil

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
	
	im = Image.open('./BodyCropData/'+img)
	x, y = im.size
	if x > 224 or y > 224:
	    im.thumbnail((224,224), Image.ANTIALIAS)
	background = Image.new('RGB', (224, 224), 'black')
	background.paste(im, (0, 0))
	background.save('./BodyCropFinal/'+img, 'JPEG')

	print('Finished: ' + img)

if __name__ == "__main__":
	start = time.time()

	def load_obj(name):
		with open(name + '.pkl', 'rb') as f:
			return pickle.load(f)

	images = [x for x in os.listdir('./BodyCropData')]
	print(len(images))

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