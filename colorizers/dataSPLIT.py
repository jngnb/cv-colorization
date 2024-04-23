import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
#split a trainset and a test set now.

def dataSplit():
    
    
    ABdata = np.load('data/true_color_downsampled/color.npy')
    Ldata = np.load('data/grayscale_downsampled/grayscale.npy')
    print(f'Ldata content:  {Ldata[0, :, :]}')
    print(f'Ldata length: {len(Ldata)}')
    print(f'ABdata length:  {len(ABdata)}')
    #ABdata = ABdata[:,:,:]

    print(ABdata.shape) # (1000, 64, 64, 2)
    print(Ldata.shape) # (1000, 64, 64)


    # print(ABdata[0, :, :, 0]==Ldata[0, :, :])
    # print(ABdata[0, :, :, 1]==Ldata[0, :, :])
    # print(ABdata[0, :, :, 2]==Ldata[0, :, :])

    #X is the features and Y is the labels

    # print(f'Ldata first row is {Ldata[0, :, :]}')
    # print(f'ABdata first row is {ABdata[0, :, :]}')

    # L_train, L_test, AB_train, AB_test = train_test_split(Ldata, ABdata, test_size=0.2, random_state=42)
    indices = np.arange(len(ABdata))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    AB_train, AB_test = ABdata[train_indices], ABdata[test_indices]
    L_train, L_test = Ldata[train_indices], Ldata[test_indices]

    print(L_train.shape)
    print(L_test.shape)
    print(AB_train.shape)
    print(AB_test.shape)





    np.save('L_train.npy', L_train)
    np.save('L_test.npy', L_test)
    np.save('AB_train.npy', AB_train)
    np.save('AB_test.npy', AB_test)

    print("successfully split")
    return L_train, L_test, AB_train, AB_test

