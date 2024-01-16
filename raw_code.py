"""Linear_Regression_Model_Practice.ipynb

### Setting up the data
- Set up the data and fit it correctly for the neural network
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# https://github.com/stedy/Machine-Learning-with-R-datasets Link to use


np.set_printoptions(precision=3, suppress=True)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

column_names = ["mpg", "cylinders", "replacement", "horsepower", "weight", "acceleration", "model_year", "origin"]

raw_data = pd.read_csv(url, names = column_names, na_values = "?",  comment='\t', sep = ' ', skipinitialspace = True)
raw_data

raw_data.describe().transpose() # Get more info with the data given

raw_data.isna().sum() #Checking if there are nan values in an rows (counting them)

raw_data = raw_data.dropna() # Getting rid of those nan values

raw_data.isna().sum()

mpg_features = raw_data.drop("mpg", axis = 1)
mpg_features

mpg_labels = raw_data["mpg"]
mpg_labels

mpg_features.head()

mpg_labels.head()

# No Encoding needed beacuse there are no non-numerical labels in the features

"""### Lets Make some Training and Testing data"""

len(mpg_features), len(mpg_labels)

mpg_train_features, mpg_test_features, mpg_train_labels, mpg_test_labels = train_test_split(mpg_features, mpg_labels, test_size = 0.2, random_state = 42)

len(mpg_train_features), len(mpg_test_features), len(mpg_train_labels), len(mpg_test_features)

mpg_train_features.head()

mpg_train_labels.head()

"""### Neural Network Regession models"""

# Sequential Model number 1

tf.random.set_seed(42)

mpg_model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
], "mpg_model_1")

mpg_model_1.compile(loss = tf.keras.losses.mae,
                        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01),
                        metrics = tf.keras.losses.mse)

mpg_model_1.fit(mpg_train_features, mpg_train_labels, epochs = 100)

mpg_model_1.summary()

tf.keras.utils.plot_model(mpg_model_1, show_shapes = True)

# Evalutaing this mpg_model_1

mpg_model_1.evaluate(mpg_test_features, mpg_test_labels)

mpg_model_1.predict(mpg_test_features.head())

print(mpg_test_labels.head())

# Improving the model
tf.random.set_seed(42)

mpg_model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(50),
    tf.keras.layers.Dense(1)
])

mpg_model_2.compile(loss = tf.keras.losses.mae,
                    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01),
                    metrics = tf.keras.losses.mse)

mpg_fit = mpg_model_2.fit(mpg_train_features, mpg_train_labels, epochs = 100)

mpg_model_2.evaluate(mpg_test_features, mpg_test_labels)

mpg_model_2.predict(mpg_test_features.head() )

mpg_test_labels.head()

tf.keras.utils.plot_model(mpg_model_2, show_shapes = True)

"""### Visualizing the Data we have so far"""

# Plot History (Loss / Training Curve)

pd.DataFrame(mpg_fit.history).plot()

plt.ylabel("loss")
plt.xlabel("epochs")

mpg_features

mpg_features["weight"].plot(kind = "hist")

mpg_features["horsepower"].plot(kind = "hist")

mpg_labels.plot(kind = "hist")

"""### Normalizing the data for better results"""

scaler = MinMaxScaler()

mpg_train_features, mpg_test_features, mpg_train_labels, mpg_test_labels = train_test_split(mpg_features, mpg_labels, test_size = 0.2, random_state = 42)
scaler.fit(mpg_train_features)

mpg_train_features_normal = scaler.transform(mpg_train_features)
mpg_test_features_normal = scaler.transform(mpg_test_features)

mpg_train_features_normal[0]

"""### Making a model on the normalized data"""

mpg_model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(50),
    tf.keras.layers.Dense(1)
])


mpg_model_3.compile(
    loss = tf.keras.losses.mae,
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01),
    metrics = tf.losses.mse
)

mpg_model_3.fit(mpg_train_features_normal, mpg_train_labels, epochs = 100)

tf.keras.utils.plot_model(mpg_model_3, show_shapes = True)

mpg_model_2_mae = mpg_model_2.evaluate(mpg_test_features, mpg_test_labels)
mpg_model_3_mae = mpg_model_3.evaluate(mpg_test_features_normal, mpg_test_labels)

mpg_model_2_mae, mpg_model_3_mae

"""### Final analysis"""

model_predictions = mpg_model_3.predict(mpg_test_features_normal).flatten()
model_predictions

test_values = mpg_test_labels
test_values

plt.scatter(test_values, model_predictions)
plt.xlabel("True Values")
plt.ylabel("Model Predictions")
limitations = [0, 50]
plt.xlim(limitations)
plt.ylim(limitations)
_ = plt.plot(limitations, limitations)

# Error disribution
error =  test_values - model_predictions
plt.hist(error, bins = 25)
plt.xlabel("Model Prediction Error")
_ = plt.ylabel('Count')

"""- mpg_model_3 was the best model I could form
- An mae (mean absolute error) of about 2.350348711013794
- An mse (mean squared error) of about 10.927210807800293
- Keep in mind I also only put the model through 100 epochs
- Specifically uses Sequential model set and the Adam optimizer with a learning rate of 0.01
- Includes 4 layers (excluding input layer)
"""
