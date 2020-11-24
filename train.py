# imports
import sys
import os
import numpy as np
import pandas as pd
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# preparing output folder
if (not(os.path.isfile("report"))):
    os.mkdir("report")

# reading data
df = pd.read_csv('data/heart.csv', sep=',')
df_X, df_y = np.split(df, [13], axis=1)

# preprocessing data
df_X["age"] = df_X["age"] / df_X["age"].max()
df_X["cp"] = df_X["cp"] / df_X["cp"].max()
df_X["trestbps"] = df_X["trestbps"] / df_X["trestbps"].max()
df_X["chol"] = df_X["chol"] / df_X["chol"].max()
df_X["thalach"] = df_X["thalach"] / df_X["thalach"].max()
df_X["oldpeak"] = df_X["oldpeak"] / df_X["oldpeak"].max()
df_X["slope"] = df_X["slope"] / df_X["slope"].max()
df_X["ca"] = df_X["ca"] / df_X["ca"].max()
df_X["thal"] = df_X["thal"] / df_X["thal"].max()

# generating batches
X_train, X_test, y_train, y_test = train_test_split(df_X,df_y, test_size=0.1)

# model structuring
model = keras.Sequential() 
model.add(keras.layers.Dense(units=13, input_shape=[13]))
model.add(keras.layers.Dense(units=6))
model.add(keras.layers.Dense(units=1))
model.add(keras.layers.Activation('sigmoid'))

# hyper parameters
opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=keras.optimizers.get(opt), loss='mean_squared_error')

# printing summary
stdout = sys.stdout
sys.stdout = open("report/structure.txt", "w")
model.summary()
sys.stdout = stdout

# training
history = model.fit(X_train, y_train, epochs=1000, verbose=2)

# evaluation
losses = history.history['loss']
epochs = history.epoch

plt.plot(epochs[50:], losses[50:])
plt.xlabel("epochs")
plt.ylabel('losses')
plt.title('Loss per Epoch')
plt.show()
plt.savefig('report/history.png')
fmetrics = open("report/metrics.txt", "w")
print('Initial loss value : ', losses[0], file=fmetrics)
print('Final loss value : ', losses[-1], file=fmetrics)

# saving
model.save('report/weights.h5')
