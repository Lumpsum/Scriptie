import pickle
import numpy as np
import os
import PIL
from PIL import Image
import tensorflow as tf
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from tensorflow.python.client import device_lib
from sklearn.metrics import precision_score
from tensorflow.python.keras import backend as K

print(device_lib.list_local_devices())

def generator(indices, batchSize = 128, training=True):

  while True:
      if training:
        np.random.shuffle(indices)
      
      for i in range(0,len(indices), batchSize):
          X = []
          real = []
          build = []
          eye = []
          hair = []
          race = []
          batch_indices = indices[i:i+batchSize]
          if training:
            batch_indices.sort()
          
          for x in batch_indices:
              X.append(np.array(Image.open('./Data2/' + x))/255)
              real.append(labels[x])
              build.append(labels2[x][:4])
              eye.append(labels2[x][4:10])
              hair.append(labels2[x][10:16])
              race.append(labels2[x][16:])
      
          X = np.array(X)
          real = np.array(real)
          build = np.array(build)
          eye = np.array(eye)
          hair = np.array(hair)
          race = np.array(race)
          yield (X, {'real':real, 'build':build, 'eye':eye, 'hair':hair, 'race':race})

def load_obj(name):
  with open(name + '.pkl', 'rb') as f:
      return pickle.load(f)

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

batchSize = 64
labels = load_obj('labelsReal2')
labels2 = load_obj('labelsCat2')
partition = {}
partition['test'] = load_obj('test')

generator_test = generator(partition['test'], batchSize, False)
print("Created generators")

json_file = open('./Architecture/VGG.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
print("Loaded model from disk")

optimizer = Adam(lr=1e-5)
loss = {'real': 'mean_squared_error', 'build': 'categorical_crossentropy',
                  'eye': 'categorical_crossentropy', 'hair': 'categorical_crossentropy',
                  'race': 'categorical_crossentropy'}
metrics = ['mse', 'accuracy']

#scores = load_obj('scores')

buildTrue = []
eyeTrue = []
hairTrue = []
raceTrue = []

for key in partition['test']:
    classes = labels2[key]
    buildTrue.append(classes[:4])
    eyeTrue.append(classes[4:10])
    hairTrue.append(classes[10:16])
    raceTrue.append(classes[16:])
    
buildTrue = [list(x).index(max(x)) for x in buildTrue]
eyeTrue = [list(x).index(max(x)) for x in eyeTrue]
hairTrue = [list(x).index(max(x)) for x in hairTrue]
raceTrue = [list(x).index(max(x)) for x in raceTrue]

for x in os.listdir('./Weights/VGG/VGG/'):
  loaded_model = model_from_json(loaded_model_json)
  loaded_model.load_weights('./Weights/VGG/VGG/'+x)
  probabilities = loaded_model.predict_generator(generator=generator(partition['test'], batchSize, False), steps=len(partition['test'])/batchSize, verbose=1)
  ap1 = precision_score(buildTrue, [list(x).index(max(x)) for x in probabilities[1]], average='macro')
  ap2 = precision_score(eyeTrue, [list(x).index(max(x)) for x in probabilities[2]], average='macro')
  ap3 = precision_score(hairTrue, [list(x).index(max(x)) for x in probabilities[3]], average='macro')
  ap4 = precision_score(raceTrue, [list(x).index(max(x)) for x in probabilities[4]], average='macro')
  mean = (ap1 + ap2 + ap3 + ap4) / 4
  scores[x + '1'] = {'build':ap1, 'eye':ap2, 'hair':ap3, 'race':ap4, 'map':mean}
  save_obj(scores, 'scores')
  K.clear_session()