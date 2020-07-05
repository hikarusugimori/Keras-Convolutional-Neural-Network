import pandas as pd
from pandas.core.common import flatten
import requests
from bs4 import BeautifulSoup as bs
import urllib.request as urllib2
import os
from clint.textui import progress, puts, colored
import os
import numpy as np
from clint.textui import progress, puts, colored
import time
import subprocess
import PIL
import glob, shutil


def aggregate_data():
    # This is the function that will take the MIT data, the AVA dataset, and the image vectorization and then combine them all.
#    df = pd.DataFrame(columns=['dp_image_id','mean_score','scene1','scene2','scene3','scene4','scene5','vector','Complementary_Colors', 'Duotones', 'HDR', 'Image_Grain', 'Light_On_White', 'Long_Exposure', 'Macro', 'Motion_Blur', 'Negative_Image', 'Rule_of_Thirds','Shallow_DOF', 'Silhouettes', 'Soft_Focus', 'Vanishing_Point'])
    df = pd.DataFrame(columns=['dp_image_id','mean_score','vector','Complementary_Colors', 'Duotones', 'HDR', 'Image_Grain', 'Light_On_White', 'Long_Exposure', 'Macro', 'Motion_Blur', 'Negative_Image', 'Rule_of_Thirds','Shallow_DOF', 'Silhouettes', 'Soft_Focus', 'Vanishing_Point'])
    image_list = pd.read_csv("AVA.csv") # Full list of images available
    files = os.listdir("image_dataset") # List of images we've downloaded and have available
    styled_images = pd.read_csv("style_multilabel.csv")
    styled_images.columns = ['dp_image_id', 'Complementary_Colors', 'Duotones', 'HDR', 'Image_Grain', 'Light_On_White', 'Long_Exposure', 'Macro', 'Motion_Blur', 'Negative_Image', 'Rule_of_Thirds','Shallow_DOF', 'Silhouettes', 'Soft_Focus', 'Vanishing_Point']
    for filename in files:
        try:
            temp_df = pd.Series(index=df.columns)
            temp_df['dp_image_id'] = filename[:-4]

            # Calculate the average score for each image
            temp_calcs = []
            temp = image_list.loc[image_list['dp_image_id']==int(filename[:-4])]
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
            temp_df['mean_score'] = avg

            # Run the image through the MIT model
    #        classifications = classify_scene(filename)
    #        try:
    #            temp_df['scene1'] = classifications[0]
    #        except:
    #            pass
    #        try:
    #            temp_df['scene2'] = classifications[1]
    #        except:
    #            pass
    #        try:
    #            temp_df['scene3'] = classifications[2]
    #        except:
    #            pass
    #        try:
    #            temp_df['scene4'] = classifications[3]
    #        except:
    #            pass
    #        try:
    #            temp_df['scene5'] = classifications[4]
    #        except:
    #            pass

            # Get the aesthetic qualities of the image
            aesthetics = styled_images.loc[styled_images['dp_image_id']==int(filename[:-4])].reset_index()
            if len(aesthetics) is 0:
                pass
            else:
                temp_df['Complementary_Colors'] = aesthetics['Complementary_Colors'][0]
                temp_df['Duotones'] = aesthetics['Duotones'][0]
                temp_df['HDR'] = aesthetics['HDR'][0]
                temp_df['Image_Grain'] = aesthetics['Image_Grain'][0]
                temp_df['Light_On_White'] = aesthetics['Light_On_White'][0]
                temp_df['Long_Exposure'] = aesthetics['Long_Exposure'][0]
                temp_df['Macro'] = aesthetics['Macro'][0]
                temp_df['Motion_Blur'] = aesthetics['Motion_Blur'][0]
                temp_df['Complementary_Colors'] = aesthetics['Complementary_Colors'][0]
                temp_df['Negative_Image'] = aesthetics['Negative_Image'][0]
                temp_df['Rule_of_Thirds'] = aesthetics['Rule_of_Thirds'][0]
                temp_df['Shallow_DOF'] = aesthetics['Shallow_DOF'][0]
                temp_df['Silhouettes'] = aesthetics['Silhouettes'][0]
                temp_df['Soft_Focus'] = aesthetics['Soft_Focus'][0]
                temp_df['Vanishing_Point'] = aesthetics['Vanishing_Point'][0]

            # Vectorize images
        except:
            pass
        #Join dataframes
        df = df.append(temp_df, ignore_index=True)
        print(df.loc[len(df)-1])
    return df

def overlap():
    files = os.listdir("image_dataset") # List of images we've downloaded and have available
    files = [filename[:-4] for filename in files if '.DS' not in filename]
    styled_images = pd.read_csv("style_multilabel.csv")
    styled_images.columns = ['dp_image_id', 'Complementary_Colors', 'Duotones', 'HDR', 'Image_Grain', 'Light_On_White', 'Long_Exposure', 'Macro', 'Motion_Blur', 'Negative_Image', 'Rule_of_Thirds','Shallow_DOF', 'Silhouettes', 'Soft_Focus', 'Vanishing_Point']
    images = styled_images['dp_image_id']
    images = images.apply(str)
    overlap = list(set(files) & set(images.tolist()))
    percentage = len(overlap)/len(images)
    return percentage, overlap

def move():
    for i in glob.glob('*.jpg'):
        print(i)
        shutil.move(i, '/Users/dbv/Documents/GitHub/places365-master/AVA_dataset/image_dataset/' + i)

def collect_aesthetic_images():
    image_list = pd.read_csv("AVA.csv")
    files = os.listdir("image_dataset")
    styled_images = pd.read_csv("style_multilabel.csv")
    styled_images.columns = ['dp_image_id', 'Complementary_Colors', 'Duotones', 'HDR', 'Image_Grain', 'Light_On_White', 'Long_Exposure', 'Macro', 'Motion_Blur', 'Negative_Image', 'Rule_of_Thirds','Shallow_DOF', 'Silhouettes', 'Soft_Focus', 'Vanishing_Point']
    styled_images['challenge'] = np.nan
    for index, row in styled_images.iterrows():
        styled_images['challenge'][index] = image_list.loc[image_list['dp_image_id']==styled_images['dp_image_id'][index]]['challenge']
    styled_images['challenge'] = pd.to_numeric(styled_images['challenge'],downcast = 'signed')
    for index, row in styled_images.iterrows():
        if files.count(str(styled_images.loc[index]['dp_image_id'])+".jpg") == 0:
            url = create_image_url(styled_images.loc[index]['dp_image_id'], styled_images.loc[index]['challenge'])
            save_photo_by_url(url, styled_images.loc[index]['dp_image_id'])
            time.sleep(4.5)
        else:
            print("Already have "+str(styled_images.loc[index]['dp_image_id']))

def collect_images(start):
    image_list = pd.read_csv("AVA.csv")
    files = os.listdir("image_dataset")
#    wks = connect_to_gsheets()
    for index, row in image_list.iterrows():
        if index >= start:
            if files.count(str(image_list.loc[index]['dp_image_id'])+".jpg") == 0:
                url = create_image_url(image_list.loc[index]['dp_image_id'], image_list.loc[index]['challenge'])
                save_photo_by_url(url, image_list.loc[index]['dp_image_id'])
#                current = wks.find(str(str(image_list.loc[index]['dp_image_id'])))
#                wks.update_acell('B'+str(current.row),1)
                time.sleep(6.5)
            else:
                print("Already have "+str(image_list.loc[index]['dp_image_id']))
        else:
            pass

def connect_to_gsheets():
    scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_name('boothml-tracker-3ff73e7207eb.json', scope)
    gc = gspread.authorize(credentials)
    wks = gc.open("MLTracker").sheet1
    return wks


def create_image_url(dp_image_id, challenge):
    if challenge <= 999:
        photo_range = "0-999"
    elif 1000 <= challenge <= 1999:
        photo_range = "1000-1999"
    elif 2000 <= challenge <= 2999:
        photo_range = "2000-2999"
    elif 3000 <= challenge <= 3999:
        photo_range = "3000-3999"
    elif 4000 <= challenge <= 4999:
        photo_range = "4000-4999"
    elif 5000 <= challenge <= 5999:
        photo_range = "5000-5999"
    elif 6000 <= challenge <= 6999:
        photo_range = "6000-6999"
    elif 7000 <= challenge <= 7999:
        photo_range = "7000-7999"
    elif 8000 <= challenge <= 8999:
        photo_range = "8000-8999"
    elif 9000 <= challenge <= 9999:
        photo_range = "9000-9999"
    url = 'http://images.dpchallenge.com/images_challenge/'+photo_range+'/' + \
        str(challenge)+'/1200/Copyrighted_Image_Reuse_Prohibited_' + \
            str(dp_image_id)+'.jpg'
    return url


def save_photo_by_url(url, dp_image_id):
    fname = 'image_dataset/' + str(dp_image_id) + '.jpg'
    try:
        with open(fname, 'wb') as f:
            f.write(requests.get(url).content)
        puts(colored.green("Finished downloading " + fname))
    except Exception as e:
        print('ERROR: {}'.format(e))


def classify_scene(image):
    # This is extremely janky and does not run without docker already running in your terminal.
    # Please install Docker Desktop on your computer and be in the "AVA_dataset" directory when in the terminal.
    try:
        out = subprocess.check_output("docker run places365_container python run_scene.py image_dataset/"+image, shell=True, universal_newlines=True)
        scene_ideas = [item[2:] for item in out.splitlines()]
        return scene_ideas
    except Exception as e:
        try:
            process = subprocess.run('docker build -t places365_container .', shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)
            out = subprocess.check_output("docker run places365_container python run_scene.py image_dataset/"+image, shell=True, universal_newlines=True)
            scene_ideas = [item[2:] for item in out.splitlines()]
            return scene_ideas
        except:
            print('ERROR: {}'.format(e))
