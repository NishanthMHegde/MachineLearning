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