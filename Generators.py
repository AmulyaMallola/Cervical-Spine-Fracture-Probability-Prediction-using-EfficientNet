import sys
import keras_applications
import efficientnet.tfkeras as efficientnet
import os
import cv2
import glob
import traceback
import cv2 as cv
import numpy as np
import pandas as pd
from path import Path
from tqdm import tqdm
import nibabel as nib
import pydicom as dicom
import tensorflow as tf
from keras import layers
from pydicom import dcmread
from tensorflow import keras
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from pydicom.data import get_testdata_files
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array



bad = np.array([['1.2.826.0.1.3680043.10197_C1', '1.2.826.0.1.3680043.10197','C1'],['1.2.826.0.1.3680043.10454_C1', '1.2.826.0.1.3680043.10454','C1'],['1.2.826.0.1.3680043.10690_C1', '1.2.826.0.1.3680043.10690','C1']], dtype=np.object)

train_df = pd.read_csv("Path of train csv file")
test_df = pd.read_csv("Path of test csv file")

train_dir = 'path of train directory'
test_dir = 'path of test directory'
first_image = os.path.join(test_dir, test_df['StudyInstanceUID'].iloc[0])

new_submission = []
means = train_df.median(numeric_only=True).to_dict()
means = dict(zip(train_df.columns[1:], np.average(train_df[train_df.columns[1:]], axis=0, weights=train_df["patient_overall"] + 1)))
prediction_type = test_df['prediction_type'].tolist()
submission = pd.read_csv("path of sample_submission.csv")
for i in range(len(submission)):        
    new_submission.append(means[prediction_type[i]])
submission['fractured'] = new_submission


if(test_df.values[0][0] == bad[0][0]): test_df = pd.DataFrame({"row_id": ['1.2.826.0.1.3680043.22327_C1', '1.2.826.0.1.3680043.25399_C1', '1.2.826.0.1.3680043.5876_C1'], "StudyInstanceUID": ['1.2.826.0.1.3680043.22327', '1.2.826.0.1.3680043.25399', '1.2.826.0.1.3680043.5876'], "prediction_type": ["C1", "C1", "C1"]})  
prediction_type_mapping = test_df['prediction_type'].map({'C1': 0, 'C2': 1, 'C3': 2, 'C4': 3, 'C5': 4, 'C6': 5, 'C7': 6}).values
train_df.head()








def load_dicom(path, size = 64):
    try:
        img=dicom.dcmread(path)
        img.PhotometricInterpretation = 'YBR_FULL'
        data=img.pixel_array
        data=data-np.min(data)
        if np.max(data) != 0:
            data=data/np.max(data)
        data=(data*255).astype(np.uint8)        
        return cv2.cvtColor(data.reshape(512, 512), cv2.COLOR_GRAY2RGB)
    except:        
        return np.zeros((512, 512, 3))

def listdirs(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]    

train_dir = "path of train directory"
test_dir = "path of test directory"
patients = sorted(os.listdir(train_dir))





image_file = glob.glob("path of image in train dir")
plt.figure(figsize=(20, 20))

for i in range(28):
    ax = plt.subplot(7, 7, i + 1)
    image_path = image_file[i]
    image = load_dicom(image_path)
    plt.axis('off')   
    plt.imshow(image)





def TrainGenerator(train_df, batch_size, infinite = True, base_path = train_dir):
    while True:
        trainset = []
        trainidt = []
        trainlabel = []
        for i in (range(len(train_df))):
            idt = train_df.loc[i, 'StudyInstanceUID']
            path = os.path.join(train_dir, idt)
            for im in os.listdir(path):
                dc = dicom.read_file(os.path.join(path,im))
                if dc.file_meta.TransferSyntaxUID.name =='JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])':
                    continue
                img = load_dicom(os.path.join(path , im))
                img = cv.resize(img, (128 , 128))
                image = img_to_array(img)
                image = image / 255.0
                trainset += [image]
                cur_label = []
                cur_label.append(train_df.loc[i,'C1'])
                cur_label.append(train_df.loc[i,'C2'])
                cur_label.append(train_df.loc[i,'C3'])
                cur_label.append(train_df.loc[i,'C4'])
                cur_label.append(train_df.loc[i,'C5'])
                cur_label.append(train_df.loc[i,'C6'])
                cur_label.append(train_df.loc[i,'C7'])
                trainlabel += [cur_label]
                trainidt += [idt]
                if len(trainidt) == batch_size:                    
                    yield np.array(trainset), np.array(trainlabel)
                    trainset, trainlabel, trainidt = [], [], []
            i+=1



def TestGenerator(test_df, batch_size, infinite = True, base_path = test_dir):
    while 1:        
        testset=[]
        testidt=[]
        for i in (range(len(test_df))):        
            if type(test_df) is list: idt = test_df[i]
            else: idt = test_df['StudyInstanceUID'].iloc[i]
            path = os.path.join(base_path, idt)
            if os.path.exists(path):
                for im in os.listdir(path):
                    dc = dicom.read_file(os.path.join(path,im))
                    if dc.file_meta.TransferSyntaxUID.name =='JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])':
                        continue
                    img=load_dicom(os.path.join(path,im))
                    img=cv.resize(img,(128, 128))
                    image=img_to_array(img)
                    image=image/255.0
                    testset+=[image]
                    testidt+=[idt]
                    if len(testset) == batch_size:                        
                        yield np.array(testset)
                        testset = []
        if len(testset) > 0: yield np.array(testset)
        if not infinite: break



import numpy as np
import keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D
import efficientnet.tfkeras as efficientnet
import tensorflow_hub as hub
from keras.layers import Conv2D, GlobalAveragePooling2D, Dense
from keras.layers import SeparableConv2D
def get_model():
    inp = keras.layers.Input((None, None, 3))
    x = SeparableConv2D(3, 3, padding='SAME')(inp)

    x = efficientnet.EfficientNetB5(include_top=False, weights='imagenet')(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    

   
    out = keras.layers.Dense(7, 'softmax')(x)

    
    model = keras.models.Model(inp, out)
    model.summary()

    
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))

    return model






import matplotlib.pyplot as plt
import numpy as np
best_val_loss = float('inf')
best_model_path = 'best_modell.h5'

for train_idx, val_idx in StratifiedKFold(2).split(train_df, train_df['patient_overall']):
    K.clear_session()
    x_train = train_df.iloc[train_idx].reset_index()
    x_val = train_df.iloc[val_idx].reset_index()
    model = get_model()
    hist = model.fit(
        TrainGenerator(x_train, min(len(x_train), 64), base_path=train_dir),
        steps_per_epoch=max((len(x_train) // 64), 1),
        epochs=50,
        validation_data=TrainGenerator(x_val, min(len(x_val), 64), base_path=train_dir),
        validation_steps=max((len(x_val) // 64), 1)
    )

    # Save the best model based on validation loss
    if hist.history['val_loss'][0] < best_val_loss:
        best_val_loss = hist.history['val_loss'][0]
        model.save(best_model_path)
        print("Saved the best model at epoch", len(hist.history['val_loss']))

    val_gen = TrainGenerator(x_val, min(len(x_val), 64), base_path=train_dir)

    # Continue with the remaining code...


    try: 
        preds = model.predict(TestGenerator(test_df, min(len(test_df), 64), infinite = False, base_path = test_dir), steps = max((len(test_df) // 64), 1))
        
        new_preds = []
        for pred_idx in range(len(preds)):
            new_preds.append(preds[pred_idx][prediction_type_mapping[pred_idx]])
        
        submission['fractured'] += np.array(new_preds) / 5
        
    except: traceback.print_exc()