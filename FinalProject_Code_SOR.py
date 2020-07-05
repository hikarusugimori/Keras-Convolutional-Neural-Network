import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pandas.core.common import flatten
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sn
from PIL import Image
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow
import os
import pickle
import time
from aesthetic_model import structure
from sklearn import preprocessing
from keras.optimizers import Adam

def collect_models():
    final_df = pickle.load(open('final_df.pkl','rb'))
    encoder = pickle.load(open('encoder_trained.pkl','rb'))
    char_model = pickle.load(open('characteristic_model_trained.pkl','rb'))
    score_model = pickle.load(open('score_model_trained.pkl','rb'))
    return encoder, char_model, score_model, final_df

def develop_characteristic_prediction(vectors, labels):
    # Split model
    Xtrain, Xtest, ytrain, ytest = train_test_split(vectors, labels, random_state=0, test_size=0.1)

    # Create model
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(300,300,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(14, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(Xtrain, ytrain, epochs=5, validation_data=(Xtest, ytest), batch_size=64)
    return model

def develop_decompressor(vectors, labels):
    # Set up autoencoding model
    x_train, x_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.3, random_state=0)
    encoding_dim = 500
    size = (vectors.shape[1]*vectors.shape[2]*vectors.shape[3])
    input_img = Input(shape=(size,))
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    decoded = Dense(size, activation='sigmoid')(encoded)
    autoencoder = Model(input_img, decoded) # Reconstruction model
    encoder = Model(input_img, encoded) # Only encoder model
    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input)) # Decoder model
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    autoencoder.fit(x_train, x_train,
                epochs=5, # TODO: Increase later
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
    autoencoder.save_weights('autoencoder_classification.h5')
    return autoencoder, encoder, decoder

def predict_score_for_image(image_file, encoder, char_model, score_model):
    # First get the image
    im = Image.open(image_file)
    im = im.resize((300,300))
    im = np.asarray(im)
    # Then encode into our autoencoder that compresses stuff
    vectors = im.astype(object)
    vectors = np.expand_dims(vectors,axis=0)
    characteristics = char_model.predict(vectors)
    # Get the output, put into an array that has 514 characteristics
    vectors = vectors.astype('float32') / 255.
    vectors = vectors.reshape((len(vectors), np.prod(vectors.shape[1:])))
    encoded_imgs = encoder.predict(vectors)
    # Run through our model and spit out prediction
    model_input = np.concatenate((encoded_imgs,characteristics),axis=None)
    model_input = np.array([model_input])
    score = score_model.predict(model_input)
    return score

def score_prediction_dataset(df):
    # Reading the CSV
    # Splitting our data into features and labels, with features as x and labels as y
    x_df  = df.drop(['File', 'Avg Score'], axis=1)
    y_df = df['Avg Score']
    y_df = y_df.divide(10)
    values = df['File']
    x = x_df.values
    y = y_df.values
    # Scaling the data so that the input features will have similar orders of magnitude
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scale = min_max_scaler.fit_transform(x)
    # Splitting our data into a training set, a validation set and a test set
    x_train, x_test_val, y_train, y_test_val = train_test_split(x_scale, y, test_size=0.4)
    x_val, x_test, y_val, y_test = train_test_split(x_test_val, y_test_val, test_size=0.5)
    return x_train, x_val, x_test, y_train, y_test, y_val, values, x_scale

def score_prediction_model(x_train, x_val, x_test, y_train, y_test, y_val, x_scale, final_df):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(x_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    # Selecting Optimizer
    optimizer = Adam(0.0002, 0.5)
    model.compile(optimizer=optimizer,
              loss='mean_squared_error',
              metrics=['accuracy'])
    hist = model.fit(x_train, y_train, batch_size=32, epochs=200, validation_data=(x_val, y_val))
    results = model.predict(x_scale)
    results = pd.DataFrame(results)
    results[0] = results[0]*10
    results['dp_image_id'] = final_df['File']
    results['original'] = final_df['Avg Score']
    results = results[['dp_image_id','original',0]]
    results.rename({0:"predicted"})
    return model, round(results,2)

def normalize_images_from_spencer(vectors,prediction):
    image_list = pd.read_csv("AVA.csv") # Full list of images available
    final_df = pd.DataFrame(vectors)
    final_df['Avg Score'] = np.nan
    for index, row in vectors.iterrows():
        try:
            temp_calcs = []
            temp = image_list.loc[image_list['dp_image_id']==int(vectors['File'][index][:-4])]
            temp_calcs.append([1] * temp.iloc[0]['1'])
            temp_calcs.append([2] * temp.iloc[0]['2'])
            temp_calcs.append([3] * temp.iloc[0]['3'])
            temp_calcs.append([4] * temp.iloc[0]['4'])
            temp_calcs.append([5] * temp.iloc[0]['5'])
            temp_calcs.append([6] * temp.iloc[0]['6'])
            temp_calcs.append([7] * temp.iloc[0]['7'])
            temp_calcs.append([8] * temp.iloc[0]['8'])
            temp_calcs.append([9] * temp.iloc[0]['9'])
            temp_calcs.append([10] * temp.iloc[0]['10'])
            temp_calcs = list(flatten(temp_calcs))
            avg = np.mean(temp_calcs)
            final_df['Avg Score'][index] = avg
            final_df['Classification 1'] = prediction.loc[index][0]
            final_df['Classification 2'] = prediction.loc[index][1]
            final_df['Classification 3'] = prediction.loc[index][2]
            final_df['Classification 4'] = prediction.loc[index][3]
            final_df['Classification 5'] = prediction.loc[index][4]
            final_df['Classification 6'] = prediction.loc[index][5]
            final_df['Classification 7'] = prediction.loc[index][6]
            final_df['Classification 8'] = prediction.loc[index][7]
            final_df['Classification 9'] = prediction.loc[index][8]
            final_df['Classification 10'] = prediction.loc[index][9]
            final_df['Classification 11'] = prediction.loc[index][10]
            final_df['Classification 12'] = prediction.loc[index][11]
            final_df['Classification 13'] = prediction.loc[index][12]
            final_df['Classification 14'] = prediction.loc[index][13]
            print(index)
        except:
            pass
    final_df.dropna(inplace=True)
    return final_df
