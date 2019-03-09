import time
import flickrapi
from queue import Queue
from threading import Thread
import os
from PIL import Image
import pickle
import shutil
import string

class DownloadWorker(Thread):
	def __init__(self, queue):
		Thread.__init__(self)
		self.queue = queue

	def run(self):
		while True:
			# Retrieve from queue and perform tasks
			img = self.queue.get()
			copy(img)
			self.queue.task_done()

def copy(x):
	shutil.copy2('./Data2/' + x, './TestData/'+x)

	print('Finished: ' + x)

if __name__ == "__main__":
	start = time.time()

	def load_obj(name):
		with open(name + '.pkl', 'rb') as f:
			return pickle.load(f)

	test = load_obj('./Pickles/test')

	# Create a queue to communicate with the worker threads
	queue = Queue()
	# Create 8 worker threads
	for x in range(100):
		worker = DownloadWorker(queue)
		# Setting daemon to True will let the main thread exit even though the workers are blocking
		worker.daemon = True
		worker.start()
	# Put the tasks into the queue as a tuple
	for x in test:
		queue.put(x)
	# Causes the main thread to wait for the queue to finish processing all the tasks
	queue.join()

print("done in : ", time.time()-start)