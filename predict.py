import pickle
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.models import model_from_json
import numpy as np
from PIL import Image

def generator(indices, batchSize = 128):

    while True:

        for i in range(0,len(indices), batchSize):
            X = []
            Y = []
            Z = []
            batch_indices = indices[i:i+batchSize]

            for x in batch_indices:
                X.append(np.array(Image.open('./Data2/' + x))/255)
                Y.append(labels[x])
                Z.append(labels2[x])

            X = np.array(X)
            Y = np.array(Y)
            Z = np.array(Z)
            yield (X, {'dense_3':Y, 'dense_4':Z})

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

batchSize = 16
labels = load_obj('./Pickles/labelsReal')
labels2 = load_obj('./Pickles/labelsCat')
partition = {}
partition['train'] = load_obj('./Pickles/train')
partition['test'] = load_obj('./Pickles/test')
generator_test = generator(partition['test'], batchSize)

# load json and create model
optimizer = Adam(lr=1e-5)
loss = {'dense_3': 'mean_squared_error', 'dense_4': 'binary_crossentropy'}
metrics = ['mse', 'accuracy']

json_file = open('./Architecture/VGGOutputs.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model3 = model_from_json(loaded_model_json)
# load weights into new model
loaded_model3.load_weights("./Weights/VGGOutputs/weights-11-0.31.hdf5")
print("Loaded model from disk")
loaded_model3.compile(optimizer=optimizer, loss=loss, metrics=metrics)

probabilities = loaded_model3.predict_generator(generator=generator_test, steps=len(partition['test'])/batchSize, verbose=1)

yPred = []

for x in probabilities[1]:
    yPred.append(np.argmax(x[:6]))
    yPred.append(np.argmax(x[6:12]) + 6)
    yPred.append(np.argmax(x[12:18]) + 12)
    yPred.append(np.argmax(x[18:]) + 18)

save_obj(yPred, 'predictions')
save_ojb(probabilities, 'probabilities')