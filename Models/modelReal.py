import pickle
import numpy as np
import os
import PIL
from PIL import Image
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

def generator(indices, batchSize = 128):

  while True:
      np.random.shuffle(indices)
      
      for i in range(0,len(indices), batchSize):
          X = []
          Y = []
          batch_indices = indices[i:i+batchSize]
          batch_indices.sort()
          
          for x in batch_indices:
              X.append(np.array(Image.open('./Data/' + x))/255)
              Y.append(labels[x])
      
          X = np.array(X)
          Y = np.array(Y)
          yield (X, Y)

def load_obj(name):
  with open(name + '.pkl', 'rb') as f:
      return pickle.load(f)

labels = {}
partition = {}
batchSize = 128
labels = load_obj('labelsReal')
partition['train'] = load_obj('train')
partition['test'] = load_obj('test')

generator_train = generator(partition['train'], batchSize)
generator_test = generator(partition['test'], batchSize)
print("Created generators")

json_file = open('./Architecture/modelReal.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./Weights/modelReal.h5")
print("Loaded model from disk")

optimizer = Adam(lr=1e-5)
loss = ['mean_squared_error']
metrics = ['mae', 'accuracy']
loaded_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

filepath = "./Weights/ModelReal/weights-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, save_best_only=False, monitor='val_loss', mode='min', verbose=1, save_weights_only=True, period=1)

tensorboard = TensorBoard(log_dir='./logs/Model2', write_graph=True, write_images=True, batch_size=batchSize)

callback_list = [checkpoint, tensorboard]

history = loaded_model.fit_generator(generator=generator_train,
                                  epochs=10,
                                  steps_per_epoch=len(partition['train'])/batchSize,
                                  validation_data=generator_test,
                                  validation_steps=len(partition['test'])/batchSize,
                                  use_multiprocessing=False,
                                  max_queue_size=10,
                                  verbose=1,
                                  callbacks=callback_list
                                  )