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
              X.append(np.array(Image.open('./BodyCropFinal/' + x))/255)
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

batchSize = 1
labels = load_obj('labelsReal3')
labels2 = load_obj('labelsCat3')
partition = {}
partition['train'] = load_obj('train2')
partition['test'] = load_obj('test2')

generator_train = generator(partition['train'], batchSize, True)
generator_test = generator(partition['test'], batchSize, False)
print("Created generators")

json_file = open('./Architecture/VGG.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./Weights/VGG.h5")
print("Loaded model from disk")

optimizer = Adam(lr=1e-5)
loss = {'real': 'mean_squared_error', 'build': 'categorical_crossentropy',
                  'eye': 'categorical_crossentropy', 'hair': 'categorical_crossentropy',
                  'race': 'categorical_crossentropy'}
metrics = ['mse', 'accuracy']
loaded_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

filepath = "./Weights/BodyCrop/weights-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, save_best_only=False, monitor='val_loss', mode='min', verbose=1, save_weights_only=False, period=1)

tensorboard = TensorBoard(log_dir='./logs/BodyCrop', write_graph=True, batch_size=batchSize)

callback_list = [checkpoint, tensorboard]

history = loaded_model.fit_generator(generator=generator_train,
                                  epochs=40,
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

