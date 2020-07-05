import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sn
from scraper import overlap
from PIL import Image
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import pickle

# If you are just starting, this is the code to run.
def main():
    # This takes the images with aesthetic qualities in the folder and vectorizing them into arrays.
    current_styled, vectors = structure()
    # This takes those vectorized images and creates an autoencoder with them.
    encoder = auto_encoding(vectors,current_styled.loc[current_styled['vectorized']==True])
    # Now that we have our model, we need to normalize the images we want to use the model to predict.
    image_list = pd.read_csv("AVA.csv") # Full list of images available
    images_to_predict, images_to_predict_vectors = normalize_images(image_list, current_styled)
    return encoder, images_to_predict, images_to_predict_vectors

def predict_features(encoder, images_to_predict, images_to_predict_vectors):
    pass

# Adapt the below
def vectorization_overlap():
    files = os.listdir("image_dataset") # List of images we've downloaded and have available
    files = [filename[:-4] for filename in files if '.DS' not in filename]
    styled_images = pd.read_csv("style_multilabel.csv")
    styled_images.columns = ['dp_image_id', 'Complementary_Colors', 'Duotones', 'HDR', 'Image_Grain', 'Light_On_White', 'Long_Exposure', 'Macro', 'Motion_Blur', 'Negative_Image', 'Rule_of_Thirds','Shallow_DOF', 'Silhouettes', 'Soft_Focus', 'Vanishing_Point']
    images = styled_images['dp_image_id']
    images = images.apply(str)
    overlap = list(set(files) & set(images.tolist()))
    percentage = len(overlap)/len(images)
    return percentage, overlap

def structure():
    styled_images = pd.read_csv("style_multilabel.csv")
    aesthetic_columns = ['dp_image_id', 'Complementary_Colors', 'Duotones', 'HDR', 'Image_Grain', 'Light_On_White', 'Long_Exposure', 'Macro', 'Motion_Blur', 'Negative_Image', 'Rule_of_Thirds','Shallow_DOF', 'Silhouettes', 'Soft_Focus', 'Vanishing_Point']
#    vector_columns = ['vector'+str(x) for x in list(np.arange(1,101))]
    styled_images.columns = aesthetic_columns
#    vector_columns = ['dp_image_id'] + vector_columns
#    vector_df = pd.DataFrame(columns=vector_columns)
    total, images = overlap()
    current_styled = styled_images[styled_images['dp_image_id'].isin(images)]
    vectors = []
    current_styled['vectorized'] = np.nan
    for index, row in styled_images.iterrows():
        try:
            im = vectorize_image(current_styled['dp_image_id'][index])
            if len(im.shape) == 3:
                vectors.append(im.astype(object))
                current_styled['vectorized'][index] = True
            print('Completed '+str(current_styled['dp_image_id'][index]))
        except:
            print('No completion of '+str(index))
    vectors = np.stack(vectors) #miracle worker
    assert len(vectors) == len(current_styled.loc[current_styled['vectorized']==True])
    return current_styled, vectors

def vectorize_image(image_id, resize=True):
    already_vectorized = os.listdir("vectorized_image_csvs") # List of images we've already vectorized
    if str(image_id)+'.pkl' in already_vectorized:
        im = pickle.load(open('vectorized_image_csvs/'+str(image_id)+'.pkl','rb'))
    else:
        im = Image.open('image_dataset/'+str(image_id)+'.jpg')
        if resize:
            im = im.resize((200,200))
        im = np.asarray(im)
        pickle.dump(im, open('vectorized_image_csvs/'+str(image_id)+'.pkl', "wb"))
    return im

# Take the result of current_styled from structure and use vectors as "styled_images" and current_styled.loc[current_styled['vectorized']==True] as labels. The size should be the same.
# encoder = auto_encoding(vectors,current_styled.loc[current_styled['vectorized']==True])

def auto_encoding(styled_images, labels):
    labels = labels[['Complementary_Colors', 'Duotones', 'HDR', 'Image_Grain', 'Light_On_White', 'Long_Exposure', 'Macro', 'Motion_Blur', 'Negative_Image', 'Rule_of_Thirds','Shallow_DOF', 'Silhouettes', 'Soft_Focus', 'Vanishing_Point']]
    x_train, x_test, y_train, y_test = train_test_split(styled_images, labels, test_size=0.3, random_state=0)
    encoding_dim = 200
    size = (styled_images.shape[1]*styled_images.shape[2]*styled_images.shape[3])
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
                epochs=1, # TODO: Increase later
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
    classify_train = autoencoder.fit(x_train, y_train, batch_size=256, epochs=5,verbose=1,validation_data=(x_test, y_test))
    autoencoder.save_weights('autoencoder_classification.h5')
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    n = 10  # how many digits we will display
    return autoencoder, x_train, x_test, y_train, y_test

#image_list = pd.read_csv("AVA.csv") # Full list of images available
#current_styled,vectors = structure()

def normalize_images(image_list, current_styled):
    to_predict = image_list[~image_list.dp_image_id.isin(current_styled.loc[current_styled['vectorized']==True]['dp_image_id'].tolist())]
    images = image_list['dp_image_id'].tolist()
    columns = ['dp_image_id', 'Complementary_Colors', 'Duotones', 'HDR', 'Image_Grain', 'Light_On_White', 'Long_Exposure', 'Macro', 'Motion_Blur', 'Negative_Image', 'Rule_of_Thirds','Shallow_DOF', 'Silhouettes', 'Soft_Focus', 'Vanishing_Point']
    all_images = pd.DataFrame(columns=columns)
    vectors = []
    for image in images:
        try:
            im = vectorize_image(image)
            if len(im.shape) == 3:
                vectors.append(im.astype(object))
                all_images.loc[len(all_images)-1] = image
                print('Completed '+str(image))
        except:
            pass
    vectors = np.stack(vectors)
    return all_images, vectors
