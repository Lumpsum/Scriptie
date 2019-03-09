import urllib.request
import time
from multiprocessing import Pool
import flickrapi
from queue import Queue
from threading import Thread
import os

class DownloadWorker(Thread):
   def __init__(self, queue):
       Thread.__init__(self)
       self.queue = queue

   def run(self):
       while True:
           # Retrieve from queue and perform tasks
           text = self.queue.get()
           download_images(text)
           self.queue.task_done()

def download_images(text):
	i = 0

	for photo in flickr.walk(text=text, sort='relevance'):
		url = 'https://farm'+photo.get('farm')+'.staticflickr.com/'+photo.get('server')+'/'+photo.get('id')+'_'+photo.get('secret')+'.jpg'
		while True:
			try:
				urllib.request.urlretrieve(url, 'Flickr/'+ text + '/' +text + str(i) +".jpg")
			except:
				print('retry' + url)
				continue
			break
    
		i += 1
		if i == 500:
			break

def check_dir(text):
	if not os.path.exists('/home/lumpsum/Desktop/Data Science/Scriptie/Flickr/'+text):
		os.makedirs('/home/lumpsum/Desktop/Data Science/Scriptie/Flickr/'+text)


if __name__ == "__main__":
	start = time.time()

	api_key = u'dfdcc6dbe417c37cdc9d2d4fabd690b7'
	api_secret = u'821c300dc6191853'

	flickr = flickrapi.FlickrAPI(api_key, api_secret, cache=True)
	flickr.authenticate_via_browser(perms='read')

	with open('nameListLarge2.txt', 'r') as f:
	    celebs = f.readlines()
	    
	celebs = [x.rstrip() for x in celebs]

	for c in celebs:
		check_dir(c)

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