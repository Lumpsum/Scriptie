import time
import flickrapi
from queue import Queue
from threading import Thread
import os
from PIL import Image

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
	print('Started: ' +text)
	for img in os.listdir('./FilteredFlickr/'+text):
		im = Image.open('./FilteredFlickr/'+text+'/'+img)
		if im.mode == 'RGB':
			width, height = im.size
			if width > height:
				im = im.resize((round(width/height*299), 299), Image.ANTIALIAS)
			else:
				im = im.resize((299, round(height/width*299)), Image.ANTIALIAS)
			imWidth = round(im.width/2)
			im = im.crop((imWidth-150, 0, imWidth+149, 299))
			im.save('./Data3/'+img)

	print('Finished: ' + text)

if __name__ == "__main__":
	start = time.time()

	with open('names.txt', 'r') as f:
		names = f.readlines()

	names = [x.rstrip() for x in names]

	# Create a queue to communicate with the worker threads
	queue = Queue()
	# Create 8 worker threads
	for x in range(100):
		worker = DownloadWorker(queue)
		# Setting daemon to True will let the main thread exit even though the workers are blocking
		worker.daemon = True
		worker.start()
	# Put the tasks into the queue as a tuple
	for text in names:
		queue.put(text)
	# Causes the main thread to wait for the queue to finish processing all the tasks
	queue.join()

print("done in : ", time.time()-start)