"""
A CNN is used for predicting/classifying images into differet classes or predicting the probability
of an image belonging to a class. The images of a particular class can have different orientation, 
color brightness, zoom on the object, position of the object, variety of the object ,etc.

There are 4 main steps in the CNN:

1. Convolution: In this step, we create a image array for our image which we wish to train. This image array
consists of values of pixels. If it is a black and white image then we have a 2D array with 0 for black and
1 for white. If it is a color image, then we have a 3D array, one each for red, blue and green.
we decide upon the different feature detectors which we will use in 
order to detect the important features of the image. These features can include edge detection, emboss,
sharpness, blur,etc. We then select an array of the same size as that of the feature detector and travel
across the input image array one stride at a time. If any value of the moving array macthes that of the feature
detector, we write the number of matches in the feature map. This wil create the feature map array. We have one feature 
map array for each feature detector.Relu is also applied in this step.

2. Pooling: We use max pooling feature for pooling. This step is done so as to take into account
minute variations in the image like the tilt, zoom, blur etc. We decide uopn a pool size. If pool size
is 2 then we take 2*2 box stride across each of the feature maps and select the maximum value out of the 4
(2*2=4) cells and place it into the max pooling array. This step also reduces the array size so that the
computation in the neural network becomes less CPU intensive.

3. Flattening: In this step, we take all the 2D max pooled image arrays and convert them into 1D array by taking
all the values in one row and making them into a column. In other words, we get an array with one column but multiple
rows. 

4. Full convolution: In this step, we create a dense neural network layer which will take the input as
the flattened image arrays. Here we apply Relu activation function. 

"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

#We have already pre-processed our data by creating a folder which has a training
#and testing sub-folder. Each of these folders has 2 more sub-folders which belong to 
#cats and dogs which we will be classifying

classifier = Sequential()
#First step is convolution where we will add the number of feature detectors and its shape
#input_shape is the (height pixels, width pixels, number of layers =3 (for RGB))
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

#Max Pooling
#pool_size is the size of the square which will stride through the Convoluted image
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#*******Add the below layer/lines of code to improve accuracy of the CNN classifier*****
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#*******************************************************
#Flattening
classifier.add(Flatten())

#We construct the Full Connection which has one dense layer and output layer consisting of one neuron
classifier.add(Dense(output_dim = 128, activation='relu'))
#We now add the output neuron layer of one neuron which predicts the probability of an image
#belonging to cat or dog
classifier.add(Dense(output_dim = 1, activation='sigmoid'))

#Now we compile our Neural Network
classifier.compile(optimizer='adam', loss = "binary_crossentropy", metrics=['accuracy'])

#Here we carry out feature augmentation where we generate different styles or renditions
#of the same training/test image so that we can simulate different types/quality or scenarios
#of images

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
#We fit our CNN classifierwith the images in training set which are now augmented
#We also carry out validation of our model against our test set to check accuracy

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)
print("Accuracy of our classifier can be improved by adding more Convolution and MaxPooling2D layers"
      " or by adding one more layer of dense neuron")