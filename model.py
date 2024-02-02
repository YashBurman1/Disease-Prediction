# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle 
#from sklearn import OrdinalEncoder
#from sklearn.ensemble import ExtraTreesRegressor


loaded_model = pickle.load(open("trained_model.sav","rb"))
loaded_encoder = pickle.load(open('encoder.sav',"rb"))
input_data = (0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
input_data_as_array = np.asarray(input_data)
input_data_reshaped = input_data_as_array.reshape(1,-1) 
prediction = loaded_model.predict(input_data_reshaped)
print(loaded_encoder.inverse_transform(prediction.reshape(1,-1)))