import time
from multiprocessing import Pool
import imageio
import os
import pickle
import numpy as np

start = time.time()

with open('names.txt', 'r') as f:
    x = f.readlines()

x = [x.rstrip() for x in x]

def readImages(x):
    print('Starting: ' + x)
    images = []
    for y in os.listdir('./FilteredFlickr/' + x):
        images.append(np.array(imageio.imread('./FilteredFlickr/' + x + '/'+ y)))

    with open('./ModelData/'+ x +'.pickle', 'wb') as handle:
        pickle.dump(np.array(list(images)), handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Finished: ' + x)


if __name__ == "__main__":
    p = Pool(processes=20)
    result = p.map(readImages, x[:1])

print("done in : ", time.time()-start)