import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm

GENERATE_NPY_FILES = False

if GENERATE_NPY_FILES:
    ant = np.load('ant.full.npz', allow_pickle=True, encoding='latin1')
    bee = np.load('bee.full.npz', allow_pickle=True, encoding='latin1')
    butterfly = np.load('butterfly.full.npz', allow_pickle=True, encoding='latin1')
    mosquito = np.load('mosquito.full.npz', allow_pickle=True, encoding='latin1')
    snail = np.load('snail.full.npz', allow_pickle=True, encoding='latin1')

    anttrain = np.concatenate((ant['train'], ant['valid']))
    beetrain = np.concatenate((bee['train'], bee['valid']))
    buttrain = np.concatenate((butterfly['train'], butterfly['valid']))
    mosqtrain = np.concatenate((mosquito['train'], mosquito['valid']))
    snailtrain = np.concatenate((snail['train'], snail['valid']))
    X_train = np.concatenate((ant['train'], bee['train'], butterfly['train'], mosquito['train'], snail['train']))

    antlab = np.zeros(ant['train'].shape)
    beelab = np.zeros(bee['train'].shape) + 1
    buttlab = np.zeros(butterfly['train'].shape) + 2
    mosqlab = np.zeros(mosquito['train'].shape) + 3
    snaillab = np.zeros(snail['train'].shape) + 4
    y_train = np.concatenate((antlab, beelab, buttlab, mosqlab, snaillab))
    X_test = np.concatenate((ant['test'], bee['test'], butterfly['test'], mosquito['test'], snail['test']))

    antlab = np.zeros(ant['test'].shape)
    beelab = np.zeros(bee['test'].shape) + 1
    buttlab = np.zeros(butterfly['test'].shape) + 2
    mosqlab = np.zeros(mosquito['test'].shape) + 3
    snaillab = np.zeros(snail['test'].shape) + 4
    y_test = np.concatenate((antlab, beelab, buttlab, mosqlab, snaillab))

    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    X_trainzero = np.zeros((len(X_train), 100, 3), dtype=np.float32)

    for x in tqdm(range(len(X_train))):
        for y in range(100):
            for z in range(3):
                if y >= len(X_train[x]):
                    break
                else:
                    X_trainzero[x][y][z] = X_train[x][y][z]

    y_labhot = np.zeros((len(y_train), 5))

    for i, x in enumerate(y_train):
        if x == 0:
            y_labhot[i][0] = 1
        elif x == 1:
            y_labhot[i][1] = 1
        elif x == 2:
            y_labhot[i][2] = 1
        elif x == 3:
            y_labhot[i][3] = 1
        elif x == 4:
            y_labhot[i][4] = 1

    X_testzero = np.zeros((len(X_test), 100, 3))

    for x in tqdm(range(len(X_test))):
        for y in range(100):
            for z in range(3):
                if y >= len(X_test[x]):
                    break
                else:
                    X_testzero[x][y][z] = X_test[x][y][z]

    y_tlabhot = np.zeros((len(y_test), 5))

    for i, x in enumerate(y_test):
        if x == 0:
            y_tlabhot[i][0] = 1
        elif x == 1:
            y_tlabhot[i][1] = 1
        elif x == 2:
            y_tlabhot[i][2] = 1
        elif x == 3:
            y_tlabhot[i][3] = 1
        elif x == 4:
            y_tlabhot[i][4] = 1

    np.save('X_train', X_trainzero)
    np.save('y_train', y_labhot)
    np.save('X_test', X_testzero)
    np.save('y_test', y_tlabhot)
else:
    X_train = np.load('X_train.npy', allow_pickle=True)
    y_train = np.load('y_train.npy', allow_pickle=True)
    X_test = np.load('X_test.npy', allow_pickle=True)
    y_test = np.load('y_test.npy', allow_pickle=True)
