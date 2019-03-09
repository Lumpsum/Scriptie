import time
from queue import Queue
from threading import Thread
import os
from PIL import Image
import numpy as np

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
	if np.array(Image.open('./Data/' + text)).shape != (224,224,3):
		print('Removed: '+text)
		os.remove('./Data/' + text)

if __name__ == "__main__":
	start = time.time()

	# Create a queue to communicate with the worker threads
	queue = Queue()
	# Create 8 worker threads
	for x in range(100):
		worker = DownloadWorker(queue)
		# Setting daemon to True will let the main thread exit even though the workers are blocking
		worker.daemon = True
		worker.start()
	# Put the tasks into the queue as a tuple
	for text in os.listdir('./Data/'):
		queue.put(text)
	# Causes the main thread to wait for the queue to finish processing all the tasks
	queue.join()

print("done in : ", time.time()-start)