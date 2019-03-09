from run_keras_server import app
from tensorflow.python.keras.models import model_from_json
import tensorflow as tf
import pickle

if __name__ == "__main__":
	app.run()