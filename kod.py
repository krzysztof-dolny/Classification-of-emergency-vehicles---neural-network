import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import os
import shutil
import cv2
import glob
import random
import itertools
import matplotlib.pyplot as plot
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

os.chdir('C:/Users/krzys/projekt_przejsciowy/baza')
if os.path.isdir('train/straz') is False:
    os.makedirs('train/karetka')
    os.makedirs('train/straz')
    os.makedirs('valid/karetka')
    os.makedirs('valid/straz')
    os.makedirs('test/karetka')
    os.makedirs('test/straz')
    os.makedirs('train/wojsko')
    os.makedirs('train/nieoznakowane')
    os.makedirs('train/policja')    
    os.makedirs('test/wojsko')
    os.makedirs('test/policja')
    os.makedirs('test/nieoznakowane')
    os.makedirs('valid/wojsko')
    os.makedirs('valid/policja')
    os.makedirs('valid/nieoznakowane')
    
    for c in random.sample(glob.glob('karetka*'), 35):
        shutil.move(c,'train/karetka')
    for c in random.sample(glob.glob('straz*'), 33):
        shutil.move(c,'train/straz')
    for c in random.sample(glob.glob('karetka*'), 5):
        shutil.move(c,'valid/karetka')
    for c in random.sample(glob.glob('straz*'), 5):
        shutil.move(c,'valid/straz')
    for c in random.sample(glob.glob('straz*'), 5):
        shutil.move(c,'test/straz')
    for c in random.sample(glob.glob('karetka*'),5):
        shutil.move(c,'test/karetka')
    for c in random.sample(glob.glob('nieoznakowany*'), 39):
        shutil.move(c,'train/nieoznakowane')
    for c in random.sample(glob.glob('wojsko*'),38):
        shutil.move(c,'train/wojsko')
    for c in random.sample(glob.glob('policja*'), 39):
        shutil.move(c,'train/policja')
    for c in random.sample(glob.glob('wojsko*'), 5):
        shutil.move(c,'valid/wojsko')
    for c in random.sample(glob.glob('nieoznakowany*'), 5):
        shutil.move(c,'valid/nieoznakowane')
    for c in random.sample(glob.glob('policja*'), 5):
        shutil.move(c,'valid/policja')
    for c in random.sample(glob.glob('wojsko*'),5):
        shutil.move(c,'test/wojsko')
    for c in random.sample(glob.glob('nieoznakowany*'),5):
        shutil.move(c,'test/nieoznakowane')
    for c in random.sample(glob.glob('policja*'),5):
        shutil.move(c,'test/policja')
        
os.chdir('../../')   

train_path = 'C:/Users/krzys/projekt_przejsciowy/baza/train'
valid_path = 'C:/Users/krzys/projekt_przejsciowy/baza/valid'
test_path = 'C:/Users/krzys/projekt_przejsciowy/baza/test'

def normalize_pixels(x):
    img = x.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #img = cv2.GaussianBlur(img, (3, 3), 0) # zastosowanie rozmycia Gaussa
    #img = cv2.medianBlur(img, 3) # zastosowanie rozmycia medianowego
    
    #kernel = np.ones((3,3),np.uint8)
    #img = cv2.erode(img, kernel, iterations = 2) # zastosowanie erozji
    #img = cv2.dilate(img, kernel, iterations = 2) # zastosowanie dylatacji
    
    img = img.astype('float32') / 255.0
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def plotImages(images_arr):
    fig, axes = plot.subplots(1,5, figsize=(150,150))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plot.tight_layout()
    plot.show()
    
def plot_confusion_matrix(cm, classes,
                         normalize=False,title='Confusion matrix',
                         cmap=plot.cm.Blues):
    plot.imshow(cm, interpolation='nearest', cmap=cmap)
    plot.title(title)
    plot.colorbar()
    tick_marks= np.arange(len(classes))
    plot.xticks(tick_marks,classes,rotation=45)
    plot.yticks(tick_marks,classes)
    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix,without normalization')
    print(cm)
    thresh= cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plot.text(j,i, cm[i,j],
                 horizontalalignment="center",
                 color="white" if cm[i,j] > thresh else "black")
        plot.tight_layout()
        plot.ylabel('true')
        plot.xlabel('predict')
        
train_batches = ImageDataGenerator(preprocessing_function=normalize_pixels).flow_from_directory(directory=train_path, target_size=(150,150), classes=[ 'straz', 'karetka', 'wojsko', 'nieoznakowane', 'policja'], batch_size=5,color_mode='rgb')
valid_batches = ImageDataGenerator(preprocessing_function=normalize_pixels).flow_from_directory(directory=valid_path, target_size=(150,150), classes= [ 'straz', 'karetka','wojsko','nieoznakowane','policja'] ,batch_size=2,color_mode='rgb')
test_batches = ImageDataGenerator(preprocessing_function=normalize_pixels).flow_from_directory(directory=test_path, target_size=(150,150), classes= [ 'straz', 'karetka','wojsko','nieoznakowane','policja'] ,batch_size=2, shuffle=False,color_mode='rgb')

imgs ,lables = next(train_batches)
plotImages(imgs)
print(lables)

model = Sequential([
    Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding= 'same', input_shape=(150,150,3)),
    MaxPool2D(pool_size=(2,2), strides=2),
    
    Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2,2), strides=2),
    
    Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2,2), strides=2),
    
    Flatten(),
    Dense(units=512, activation='relu'),
    Dense(units=5, activation='softmax'),
])

model.summary()
model.compile(optimizer=Adam(learning_rate= 0.0002), loss= 'categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=2)

predictions=model.predict(x=test_batches, verbose=0)

cm= confusion_matrix(y_true=test_batches.classes,y_pred=np.argmax(predictions, axis=-1))
cm_plot_labels=['straz', 'karetka','wojsko','nieoznakowane','policja']
plot_confusion_matrix(cm=cm,classes=cm_plot_labels,title='confusion matrix')










