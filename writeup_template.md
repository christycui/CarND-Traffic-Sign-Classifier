#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 39209
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. The first image shown is the first image from the training dataset. It is a picture of a speed limit 20 sign and the label for that image is 0. When we look up the name for ClassId 0, it is indeed "Speed Limit 20km/h". The second image is a bar chart showing how a count of each classification in the training set.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I shuffled the the images and labels for the training set. I also applied normalization in pre-processing, by first subtracting the min and then deviding by the range.


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)


The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

I then split the original training set into training and validation, because there was no validation set given in the original pickle files. I randomly split the training data into a training set and validation set. I did this by using train_test_split from the sklearn model_selection library.

My final training set had 31367 number of images. My validation set and test set had 7842 and 12630 number of images.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid_padding, outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid_padding, outputs 10x10x16      									|
| RELU          |                       |
| Max pooling         | 2x2 stride, valid_padding, outputs 5x5x16        |
| Flatten       |
| Fully connected		| outputs 120        									|
| RELU          |
| Fully connected   | outputs 84                          |
| RELU          |
| Fully connected   | outputs 10                          |


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used Adam optimizer with a learning rate of 0.001, batch size of 128 and 10 epochs. 

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.987
* validation set accuracy of 0.972
* test set accuracy of 0.860

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

I chose the LeNet archetecture because the input dataset is 32x32x3, similar to the MNIST dataset upon which was proven to be robust. The model is working fairly well given that the training, validation and test accuracy all improved (despite fluctuations) as the number of epochs increased. 

At first, I tried a very similar architecture but without implementing small random initial weights. Both my training and validation accuracies were extremely low (hovering below 10%). I tried a few things to fix it but eventually found out that I initialized my weights with "truncated_normal" function but forgot to include a mean of 0 and standard deviation of 0.1. As a result, my initial architecture seems to be stuck at local minimums and had terrible accuracy.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

They are plotted out in ipython cell #11.

Since the original image are all bigger than the input of our model, 32x32. In resizing the images, we lost a lot of valuable information. That's why these real-world images can be harder to classify.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Slippery Road      		| Traffic Signals   									| 
| Children Crossing			| Traffic Signals										|
| No Entry					| Priority Road											|
| 30 km/h	      		| General Caution					 				|
| Stop        			| Bicycles Crossing      							|


The model was able to correctly guess 0 of the 5 traffic signs, which gives an accuracy of 0%. This comes out very badly compared to the 86% test accuracy.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is very confident that this is a traffic signal (probability of 0.99), but the image does not contain a traffic signal. Instead, it is a slippery road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Traffic Signals   									| 
| 0.01     				| General Caution 										|
| 0.00				| Stop											|
| 0.00	      			| Wild Animals Crossing					 				|
| 0.00				    | Road Work      							|


For the second image, the model is very confident that this is a traffic signal (probability of 0.99), but the image does not contain a traffic signal. Instead, it is a children crossing sign. The top five soft max probabilities were

| Probability           |     Prediction                    | 
|:---------------------:|:---------------------------------------------:| 
| .99               | Traffic Signals                     | 
| 0.01            | Wild animals crossing                     |
| 0.00        | beware of ice/snow                      |
| 0.00              | Bicycles Crossing                 |
| 0.00            | Slippery Road                   |

For the third image, the model is very confident that this is a priority road, but it is in fact a no entry sign. The top fix soft max probabilities were 

| Probability           |     Prediction                    | 
|:---------------------:|:---------------------------------------------:| 
| .98               | Priority Road                     | 
| 0.01            | Right-of-way at the next intersection                     |
| 0.01        | Dangerous curve to the left                      |
| 0.00              | Vehicles over 3.5 metric tons prohibited                 |
| 0.00            | Traffic Signals                   |

For the fourth image, the model is fairly certain that this is a General Caution sign but it is actually a 30km/h speed limit sign. The softmax probabilities are:

| Probability           |     Prediction                    | 
|:---------------------:|:---------------------------------------------:| 
| 0.80              | General Caution                   | 
| 0.10            | Wild animals crossing                    |
| 0.10        | Road work                    |
| 0.00              | Road narrows on the right                |
| 0.00            | Bicycles crossing                 |

For the last image, the model is very certain that this is a Bicycles crossing image but it is actually a stop sign. The softmax probabilities are:


| Probability           |     Prediction                    | 
|:---------------------:|:---------------------------------------------:| 
| 1.00              | Bicycles crossing                  | 
| 0.00            | Children crossing                    |
| 0.00        | Speed limit (30km/h)                   |
| 0.00              | Speed limit (20km/h)                |
| 0.00            | Wild animals crossing                |