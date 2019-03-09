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
          yield ([X1, X2], {'real':real, 'build':build, 'eye':eye, 'hair':hair, 'race':race})

def load_obj(name):
  with open(name + '.pkl', 'rb') as f:
      return pickle.load(f)

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

batchSize = 128
labels = load_obj('labelsReal3')
labels2 = load_obj('labelsCat3')
partition = {}
partition['train'] = load_obj('train2')
partition['test'] = load_obj('test2')

generator_train = generator(partition['train'], batchSize, True)
generator_test = generator(partition['test'], batchSize, False)
print("Created generators")

json_file = open('./Architecture/VGG2Input.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./Weights/VGG2Input.h5")
print("Loaded model from disk")

optimizer = Adam(lr=1e-5)
loss = {'real': 'mean_squared_error', 'build': 'categorical_crossentropy',
                  'eye': 'categorical_crossentropy', 'hair': 'categorical_crossentropy',
                  'race': 'categorical_crossentropy'}
metrics = ['mse', 'accuracy']
loaded_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

filepath = "./Weights/BodyFaceCrop/weights-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, save_best_only=False, monitor='val_loss', mode='min', verbose=1, save_weights_only=False, period=1)

tensorboard = TensorBoard(log_dir='./logs/BodyFaceCrop', write_graph=True, batch_size=batchSize)

callback_list = [checkpoint, tensorboard]

history = loaded_model.fit_generator(generator=generator_train,
                                  epochs=20,
                                  steps_per_epoch=len(partition['train'])/batchSize,
                                  validation_data=generator_test,
                                  validation_steps=len(partition['test'])/batchSize,
                                  use_multiprocessing=False,
                                  max_queue_size=10,
                                  class_weight={'build':{0:0.73571164, 1:1.5550806, 2:22.56222132, 3:0.51192885}, 
                                  'eye':{0:5.1338314, 1:0.57062746, 2:0.41682393, 3:14.31610082, 4:1.03796737, 5:1.61189059},
                                   'hair':{0:24.60698497, 1:0.91913537, 2:0.66172109, 3:0.32181615, 4:15.59131653, 5:5.30029043}, 
                                   'race':{0:9.57402709, 1:2.27850381, 2:3.6433317, 3:1.53009415, 4:0.28339549}},
                                  verbose=1,
                                  callbacks=callback_list
                                  )

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

for x in os.listdir('./Weights/BodyFaceCrop/'):
  loaded_model = model_from_json(loaded_model_json)
  loaded_model.load_weights('./Weights/BodyFaceCrop/'+x)
  probabilities = loaded_model.predict_generator(generator=generator(partition['test'], batchSize, False), steps=len(partition['test'])/batchSize, verbose=1)
  ap1 = precision_score(buildTrue, [list(x).index(max(x)) for x in probabilities[1]], average='macro')
  ap2 = precision_score(eyeTrue, [list(x).index(max(x)) for x in probabilities[2]], average='macro')
  ap3 = precision_score(hairTrue, [list(x).index(max(x)) for x in probabilities[3]], average='macro')
  ap4 = precision_score(raceTrue, [list(x).index(max(x)) for x in probabilities[4]], average='macro')
  mean = (ap1 + ap2 + ap3 + ap4) / 4
  scores[x + 'BodyFace'] = {'build':ap1, 'eye':ap2, 'hair':ap3, 'race':ap4, 'map':mean}
  save_obj(scores, 'scores')
  K.clear_session()