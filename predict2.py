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
from tensorflow.python.keras import backend as K
from sklearn.metrics import precision_score

print(device_lib.list_local_devices())

def generator(indices, batchSize = 128, training=True):

  while True:
      if training:
        np.random.shuffle(indices)
      
      for i in range(0,len(indices), batchSize):
          X1 = []
          X2 = []
          real = []
          build = []
          eye = []
          hair = []
          race = []
          batch_indices = indices[i:i+batchSize]
          if training:
            batch_indices.sort()
          
          for x in batch_indices:
              X1.append(np.array(Image.open('./BodyCropFinal/' + x))/255)
              X2.append(np.array(Image.open('./FaceCropFinal/' + x))/255)
              real.append(labels[x])
              build.append(labels2[x][:4])
              eye.append(labels2[x][4:10])
              hair.append(labels2[x][10:16])
              race.append(labels2[x][16:])
      
          X1 = np.array(X1)
          X2 = np.array(X2)
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

labels = load_obj('labelsReal3')
labels2 = load_obj('labelsCat3')
partition = {}
partition['train'] = load_obj('train2')
partition['test'] = load_obj('test2')

batchSize = 128

generator_train = generator(partition['train'], batchSize, True)
generator_test = generator(partition['test'], batchSize, False)
print("Created generators")

scores = load_obj('scores')

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

optimizer = Adam(lr=1e-5)
loss = {'real': 'mean_squared_error', 'build': 'categorical_crossentropy',
                  'eye': 'categorical_crossentropy', 'hair': 'categorical_crossentropy',
                  'race': 'categorical_crossentropy'}
metrics = ['mse', 'accuracy']

json_file = open('./Architecture/VGG.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

for x in os.listdir('./Weights/BodyFaceCrop/'):
  model = model_from_json(loaded_model_json)
  model.load_weights('./Weights/BodyFaceCrop/'+x)
  probabilities = model.predict_generator(generator=generator(partition['test'], batchSize, False), steps=len(partition['test'])/batchSize, verbose=1)
  ap1 = precision_score(buildTrue, [list(x).index(max(x)) for x in probabilities[1]], average='macro')
  ap2 = precision_score(eyeTrue, [list(x).index(max(x)) for x in probabilities[2]], average='macro')
  ap3 = precision_score(hairTrue, [list(x).index(max(x)) for x in probabilities[3]], average='macro')
  ap4 = precision_score(raceTrue, [list(x).index(max(x)) for x in probabilities[4]], average='macro')
  mean = (ap1 + ap2 + ap3 + ap4) / 4
  scores[x + 'BodyFace'] = {'build':ap1, 'eye':ap2, 'hair':ap3, 'race':ap4, 'map':mean}
  save_obj(scores, 'scoresTest')
  K.clear_session()