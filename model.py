#import required lib
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from sklearn.utils import shuffle
import keras
from keras.layers.convolutional import Cropping2D , Convolution2D
from keras.layers import Lambda , Flatten , Dense , Dropout 
from keras.models import Sequential, Model

print ('Done importing required libraries')

#define paramters used with-in the code
def_loc_csvfile=["data/driving_log.csv" ]
def_loc_img=["data/IMG/" ]
stearings=[]
csvData=[]
correction_par=0.20

#read csv files from location and populate the data in csvData as per requirement

i=0 

for files in def_loc_csvfile:
    
    print("*****************Reading csv file " + files + "************************")
    with open(files) as csvfile:
        lineReader = csv.reader(csvfile)
        for line in lineReader:
            #split the file to get file name
            sCenterFileName=line[0].split("\\")[-1]
            sLeftFileName=line[1].split("\\")[-1]
            sRightFileName=line[2].split("\\")[-1]
            #stores the stearing measurement value
            stearCenter=float(line[3])
            #append stearing data and image file full name
            #do this for all the three images so as to increase data size 
            #adust stearing data by adding / subtracting correction value
            csvData.append((def_loc_img[i]+sCenterFileName,stearCenter,'None'))
            csvData.append((def_loc_img[i]+sLeftFileName,stearCenter+correction_par,'None'))
            csvData.append((def_loc_img[i]+sRightFileName,stearCenter-correction_par,'None'))

            # append data for performing extra actions for augmentaion , 
            # adding for brightness change , do this for only center image
            # keep stearing angle same
            csvData.append((def_loc_img[i]+sCenterFileName,stearCenter,'Brightness'))

    i+=1
    
print ("***************File Reading Completed , below are the details*****************")
print("-------Sample Length : ",len(csvData))  


#split total data in validation and traing data using train_test_split function

train_samples, validation_samples = train_test_split(csvData, test_size=0.2)

print ("----------Sample Splitted into Train and Validation with below details" )
print("No. of Train Samples:",len(train_samples))
print("No. of Validation Samples:",len(validation_samples))

# function used as generator with model
# this alos contains logic to read image from location and 
# apply brightness change if image is set for brightness change
# else its not touched
def dataGenerator(samples,batch_size):
    
    #total no of rows in samples
    num_samples = len(samples)
 


    # infinte while loop so that generator is always open / never terminated
    while True:
        
        # shuffle the sample data before its splitted in batches
        shuffle(samples)
        #loop to split data into batches       
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            stearings = []
        # loop to iterate each row in batch so that image is read and added to 
        # image array , additonal brightness change is added if required.
            for batch_sample in batch_samples:
                image=cv2.cvtColor(cv2.imread(batch_sample[0]), cv2.COLOR_BGR2RGB)
                stearing=float(batch_sample[1])
                
                if batch_sample[2].lower()=="none":
                    images.append(image)
                
                if batch_sample[2].lower()=="brightness":
                    
                    # convert to HSV so that its easy to adjust brightness
                    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
                    # randomly generate the brightness reduction factor
                    # Add a constant so that it prevents the image from being completely dark
                    rdbrightness = .15 + np.random.uniform()
                    # Apply the brightness reduction to the V channel
                    image1[:,:,2] = image1[:,:,2]*rdbrightness
                    # convert to RBG again
                    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
                    #append to images array
                    images.append(image1)
                
                #add stearing details to the stearing array
                stearings.append(stearing)

            # convert to numpy array as its accepted by keras
            X_train = np.array(images)
            
            y_train = np.array(stearings)
            yield shuffle(X_train, y_train)


# code to call the generators
batch_szie=126
model_file="model.h5"

train_generator = dataGenerator(train_samples, batch_szie)
validation_generator = dataGenerator(validation_samples, batch_szie)

#code to define the model
# using NVEDIA architecture as base model
model = Sequential()

# performing cropping to image on top 75 pixles and bottom 25 pixles to remove 
# unnecessary data
# input shape to model is 160x320x3
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
# add lamdba layer to normalize the data between 0.5 to -0.5
model.add(Lambda(lambda x: x / 255.0 - 0.5 ))
# add CNN layer with filter 5x5x24 followed by elu activation 
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="elu"))
# add drop out to overcome over-fitting
model.add(Dropout(0.3))
# add CNN layer with filter 5x5x36 followed by elu activation
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="elu"))
# add drop out to overcome over-fitting
model.add(Dropout(0.2))
# add CNN layer with filter 5x5x48 followed by elu activation
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="elu"))
# add CNN layer with filter 3x3x64 followed by elu activation
model.add(Convolution2D(64,3,3,activation="elu"))
# add CNN layer with filter 3x3x64 followed by elu activation
model.add(Convolution2D(64,3,3,activation="elu"))
# add Flatten layer
model.add(Flatten())
# add dense layer
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
# add final output layer of 1 neuron 
model.add(Dense(1))

# use MSE as loss function and optimizer as NADAM so that learing rate 
# need not be modified and taken care by itself
model.compile(loss='mse', optimizer='nadam')



#adding checkpoints so that only best model is saved 
checkpoint=keras.callbacks.ModelCheckpoint(model_file, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

# adding early stop if mode is not improving even for a bit
earlystopping= keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

# display model summary for visualization 

model.summary()

print ("......Start Training........")

# add fit layer to train using generator
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,nb_val_samples=len(validation_samples),nb_epoch=20,verbose = 1,callbacks=[checkpoint,earlystopping])


print ("......Training Completed........")
