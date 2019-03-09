from PIL import Image
import numpy as np
import flask
from flask import Flask, render_template, request
import io
from sklearn.metrics.pairwise import cosine_similarity
import os
import simplejson as json
from tensorflow.python.keras.models import model_from_json
import tensorflow as tf
import pickle

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
loaded_model = None

def load_model():
	def load_obj(name ):
		with open(name + '.pkl', 'rb') as f:
			return pickle.load(f)

	global loaded_model
	json_file = open('./static/model/VGG.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("./static/model/weights-07-5.22.hdf5")
	global graph
	graph = tf.get_default_graph()
	print("Loaded model from disk")

	global male
	global female
	male = load_obj('male')
	female = load_obj('female')
	print("Loaded objects")

print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
load_model()

def prepare_image(image):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	width, height = image.size
	if width > height:
		image = image.resize((round(width/height*224), 224), Image.ANTIALIAS)
	else:
		image = image.resize((224, round(height/width*224)), Image.ANTIALIAS)
	imWidth = round(image.width/2)
	image = image.crop((imWidth-112, 0, imWidth+112, 224))

	image = np.array(image)
	image = np.expand_dims(image, axis=0)

	# return the processed image
	return image

@app.route("/")
def index():
    return flask.render_template('index.html');

#@app.route("/suggestion/<names>")
#def suggestion(names):
#	return flask.render_template("suggestion.html", names=names);

@app.route("/suggestion")
def suggestion():
	return flask.render_template("suggestion.html", data=data);

@app.route("/predict/<gender>", methods=["POST", "GET"])
def predict(gender):
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	image = request.get_data()

	# ensure an image was properly uploaded to our endpoint
	#if flask.request.method == "POST":
	#	if flask.request.files.get("image"):
	# read the image in PIL format
	#image = flask.request.files["image"].read()
	image = Image.open(io.BytesIO(image))

	# preprocess the image and prepare it for classification
	image = prepare_image(image)

	# classify the input image and then initialize the list
	# of predictions to return to the client
	with graph.as_default():
		preds = loaded_model.predict(image)
		data["predictions"] = []

	prob = []

	for i in range(1, 5):
		b = np.zeros_like(preds[i][0])
		b[preds[i][0].argmax()] = 1
		prob = np.concatenate((prob, b), axis=0)

	prob = np.expand_dims(prob, axis=0)

	g = None
	if gender == "male":
		g = male
	else:
		g = female

	similar = cosine_similarity(prob, g.values)
	celebs = sorted(zip(similar[0], g.index.tolist()), reverse=True)[:3]
	data["predictions"] = [x[1].replace(' ', '') for x in celebs] + [x[1].title() for x in celebs]

	# indicate that the request was a success
	data["success"] = True

	# return the data dictionary as a JSON response
	#return flask.render_template("suggestion.html", value=data['predictions'])
	return json.dumps({'success':True, 'data':flask.render_template("suggestion.html", value=data['predictions'])}), 200, {'ContentType':'application/json'}
	#return json.dumps({'success':True, 'data':data['predictions']}), 200, {'ContentType':'application/json'}

# if this is the main thread of execution first load the model and
# then start the server
#if __name__ == "__main__":
#	print(("* Loading Keras model and Flask starting server..."
#		"please wait until server has fully started"))
#	load_model()
#	port = int(os.environ.get('PORT', 5000))
#	app.run(host='0.0.0.0', port=port)