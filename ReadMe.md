**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model.PNG "Model"
[image2]: ./data_plot.PNG "Data Visual"
[image3]: ./data_plot2.PNG "Data Visual"
[image4]: ./1.jpg "Training Data"
[image5]: ./2.jpg "Training Data"
[image6]: ./3.jpg "Training Data"
[image7]: ./4.jpg "Training Data"
[image8]: ./5.jpg "Training Data"
[image9]: ./6.jpg "Training Data"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py : containing the script to create and train the model
* drive.py : for driving the car in autonomous mode (not modified ) ,already provided by Udacity Team
* model.h5 : trained convolution neural network model
* output.mp4 : video of track one with one lap of autonomous driving.
* top_output.mp4 : Top view of car driving on track , to help visulaize the bheaviour of car.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing command
```sh
python drive.py model.h5
```
Note : drive.py have to be modified to reduce the speed factor depending on the workstation we are testing on. In-Case of low end model modify line 49 `set_speed = 9` set value to 4 from 9.

#### 3. Submission code is usable and readable

The model.py file contains the code for training , data augmentation and saving the convolution neural network. 
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I choose to use NVEDIA model as base model for my arichitecture after going through udacity classroom . 

My model layer starts with Cropping layer to crop the image from top and bottom, as after visualization of the images found that data at top contians horizon and it's not proividing much insights and bottom of the image contained some portion of car itself. I thought of adding this under keras layer so the same pre-processing need not to added in drive.py and its done by model when real data is fed to it through drive.py

This was followed by Lambda layer for normalization , where the data was normalized between  -0.5 to 0.5 using equation `(x / 225.0) -0.5`

This was followed by 6 convolutional layers with filters of 5x5 , followed by 3x3 filter and depths ranging from 24 to 64, this is kept as per the NVEDIA model. Used 'ELU' as activation function across CNN layers to introduce non-linerarity .

This was followed by FLATTENING and 3 Fully Connected Layers , which is followed by single output layer.
Below is the model summary captured using command `model.summary()`

![alt text][image1]

#### 2. Attempts to reduce overfitting in the model

The model contains 2 dropout layers in order to reduce overfitting (model.py lines 133,137,148). 

Data was augmented so that no stearing angle is dominating , this alos helped overfitting.
The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 52). 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

*The model used an nadam optimizer, so the learning rate need not be tunned and is taken care by optimizer
`model.compile(loss='mse', optimizer='nadam')` (model.py line 156).

*Added checkpoints for saving best model out of the epochs run 
 `ModelCheckpoint(model_file, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)` (model.py line 161 ).

*Added Stop Early callback too so that model would training if validation_loss would increase rather tahn decrease from previous run.
`EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')` (model.py line 164 )


#### 4.  Creation of the Training Set & Training Process

Initially i tried to use UDACITY training Data set provided in project , resource , but after visualizing it found that , it contained more data with 0 degree angle , so decided to caputure my own data using simulator.
To capture good driving behavior, I first recorded two laps on track one using simulator and tried to run the car in center of lane and tried not to keep stearing angle 0 degree so that variety data can be collected. I also noticed that track one was more left centered and contained more left turns than right  so i drived the track backwards twice.
Here is an example image of driving:

   1   |  2 
------------ | -------------
![alt text][image4] | ![alt text][image5]
![alt text][image6] | ![alt text][image7]
![alt text][image8] | ![alt text][image9]


Below is the visualization of the data collected after merging all , left , right , center images from the training set and adding correction of `0.15` to the stearing angle to left and right images w.r.t. center image.
Also changed brightness of center images so to reduce over-fitting and train better.

`Total Number of Samples : 97808`

1   |  2 
------------ | -------------
![alt text][image2] | ![alt text][image3]

After this whole data was distributed to training and validaton set , below no. of records used for training and validation.

`No. of Train Samples: 78246
No. of Validation Samples: 19562`

After this i trained the model keeping `Batch Size : 128` and number of  `epochs as 30` . I had already introduced checkpoint and early stopping mechanism in case validation loss increased rather and reducing. So my model trained for `9 epochs` instead.
Below are the logs printed , you can also visit `model_log.log` for detailed log.

`Epoch 1/30 78246/78246 [==============================] - 178s - loss: 0.0121 - val_loss: 0.0101`

`Epoch 2/30 78246/78246 [==============================] - 175s - loss: 0.0091 - val_loss: 0.0084`

`Epoch 3/30 78246/78246 [==============================] - 175s - loss: 0.0072 - val_loss: 0.0059`

`Epoch 4/30 78246/78246 [==============================] - 174s - loss: 0.0060 - val_loss: 0.0052`

`Epoch 5/30 78246/78246 [==============================] - 174s - loss: 0.0053 - val_loss: 0.0050`

`Epoch 6/30 78246/78246 [==============================] - 174s - loss: 0.0047 - val_loss: 0.0044`

`Epoch 7/30 78246/78246 [==============================] - 174s - loss: 0.0044 - val_loss: 0.0041`

`Epoch 8/30 78246/78246 [==============================] - 173s - loss: 0.0042 - val_loss: 0.0039`

`Epoch 9/30 78246/78246 [==============================] - 171s - loss: 0.0040 - val_loss: 0.0041`

`......Training Completed........`
