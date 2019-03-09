import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from skimage import data, io
import shutil
from queue import Queue
from threading import Thread
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

class DownloadWorker(Thread):
	def __init__(self, queue):
		Thread.__init__(self)
		self.queue = queue

	def run(self):
		while True:
			# Retrieve from queue and perform tasks
			image = self.queue.get()
			crop_images(image)
			self.queue.task_done()

def crop_images(image):
	print('Started: ' +image)

	img = skimage.io.imread(os.path.join(IMAGE_DIR, image))
	
	results = model.detect([img], verbose=1)

	i = 0

	pos = next((i for i, x in enumerate(results[i]['class_ids']) if x==1), None)
	if pos != None:
		cropped = img[results[pos]['rois'][0][0]:results[pos]['rois'][0][2],results[pos]['rois'][0][1]:results[pos]['rois'][0][3]]
		im = Image.fromarray(cropped)
		im.save("D:/D - Desktop/Scriptie/BodyCropData/"+image+".jpg")

	print('Finished: ' + image)

if _name_ == "_main_":
	# Directory to save logs and trained model
	MODEL_DIR = os.path.join(ROOT_DIR, "logs")

	# Local path to trained weights file
	COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
	# Download COCO trained weights from Releases if needed
	if not os.path.exists(COCO_MODEL_PATH):
	    utils.download_trained_weights(COCO_MODEL_PATH)

	# Directory of images to run detection on
	IMAGE_DIR = os.path.join(ROOT_DIR, "images")

	class InferenceConfig(coco.CocoConfig):
	    # Set batch size to 1 since we'll be running inference on
	    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	    GPU_COUNT = 1
	    IMAGES_PER_GPU = 1
	    IMAGE_MIN_DIM = 200

	config = InferenceConfig()
	config.display()

	# Create model object in inference mode.
	model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

	# Load weights trained on MS-COCO
	model.load_weights(COCO_MODEL_PATH, by_name=True)

	# COCO Class names
	# Index of the class in the list is its ID. For example, to get ID of
	# the teddy bear class, use: class_names.index('teddy bear')
	class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
	               'bus', 'train', 'truck', 'boat', 'traffic light',
	               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
	               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
	               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
	               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
	               'kite', 'baseball bat', 'baseball glove', 'skateboard',
	               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
	               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
	               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
	               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
	               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
	               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
	               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
	               'teddy bear', 'hair drier', 'toothbrush']

	# Load a random image from the images folder
	file_names = next(os.walk(IMAGE_DIR))[2]
	image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

	#results = model.detect([np.empty((1,1,3))], verbose=1)

	print('Pre-Detect done')

	# Create a queue to communicate with the worker threads
	queue = Queue()
	# Create 8 worker threads
	for x in range(100):
		worker = DownloadWorker(queue)
		# Setting daemon to True will let the main thread exit even though the workers are blocking
		worker.daemon = True
		worker.start()
	# Put the tasks into the queue as a tuple
	for x in os.listdir(IMAGE_DIR):
		queue.put()
	# Causes the main thread to wait for the queue to finish processing all the tasks
	queue.join()