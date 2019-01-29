# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 05:18:28 2019

@author: 90324
"""
from tensorflow import keras
import numpy as np
train_data_path = 'data/train_input.npy'
train_label_path = 'data/train_label.npy'
test_data_path = 'data/test_input.npy'

model_name = 'cnn_model.h5'
model = keras.models.load_model(model_name)
predict_data = np.load(test_data_path)
model.predict()
predict_data = predict_data/255.0
result = model.predict(predict_data)
result_label = np.argmax(result,axis=1)
#result_label = [class_names[i] for i in result_label]
#with open('result.txt',"w") as f:
       # f.write(str(result_label)) 
        
np.save('result.npy',result_label)