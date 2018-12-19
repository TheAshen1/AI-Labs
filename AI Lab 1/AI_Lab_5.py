import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils, plot_model
from keras.callbacks import EarlyStopping

#model = Sequential() 
 
#model.add(Dense(2, input_dim=1))
#model.add(Dense(8, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))

##plot_model(model, to_file='model.png', show_shapes=True)

#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

#model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1) 

   
#early_stopping=EarlyStopping(monitor='value_loss')   
 
#model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1, callbacks=[early_stopping]) 


 
np.random.seed(7) 
# load dataset 
dataset = np.loadtxt("iris.data", delimiter=",") 
# split into input (X) and output (Y) variables 
X = dataset[:,0:4] 
Y = dataset[:,4] 
# create model 
model = Sequential() 
model.add(Dense(2, input_dim=4, activation='linear')) 
model.add(Dense(24, activation='relu')) 
model.add(Dense(1, activation='sigmoid')) 
# Compile model 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
# Fit the model 
model.fit(X, Y, epochs=100, batch_size=2) 
# evaluate the model 
scores = model.evaluate(X, Y) 
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)) 

model.predict(X)