Use np.load to load the '.npy' file. You will get a 4-dim numpy array(X*256*256*3). There are 405 images for training and 46 images for testing. 

The data format is rgb and data type is uint8 (from 0 to 255). If you did any preprocessing to the training data, you need to do the same processing to the test data!!!!

There is only input data, no lable for the test data. Save your predicted label (use 0-10, not one-hot format) and we will calculate the accuracy for you. 


 


