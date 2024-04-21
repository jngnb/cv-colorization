import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
#split a trainset and a test set now.

def dataSplit():
    ABdata = np.load('../data/true_color_downsampled/color.npy')
    Ldata = np.load('../data/grayscale_downsampled/grayscale.npy')
    print(ABdata.shape)
    print(Ldata.shape)

    #X is the features and Y is the labels
    L_train, L_test, AB_train, AB_test = train_test_split(Ldata, ABdata, test_size=0.2, random_state=42)

    np.save('L_train.npy', L_train)
    np.save('L_test.npy', L_test)
    np.save('AB_train.npy', AB_train)
    np.save('AB_test.npy', AB_test)
    return L_train, L_test, AB_train, AB_test

