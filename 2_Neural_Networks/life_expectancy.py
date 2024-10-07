import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# -----------------
# DATA LOADING & OBSERVING
# -----------------
dataset  = pd.read_csv('life_expectancy.csv')
print(dataset.head())
print(dataset.describe())

# general pattern from all countries
dataset = dataset.drop(['Country'], axis = 1)
print(dataset.head())

labels = dataset.iloc[:, -1]
features = dataset.iloc[:,0:-1]

# -----------------
# DATA PROCESSING
# -----------------
# categorical columns need to be converted into numerical columns = one-hot-encoding
features = pd.get_dummies(features)

# split data: traininig, test.
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.20, random_state=23)

# standardize/normalize
numerical_features = features.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_features.columns

ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder='passthrough')

# training data
features_train_scaled = ct.fit_transform(features_train)
# test data
features_test_scaled = ct.fit_transform(features_test)

# -----------------
# BUILDING THE MODEL
# -----------------
my_model = Sequential()

input = InputLayer(input_shape = (features.shape[1], ))
my_model.add(input)
my_model.add(Dense(64, activation = "relu"))
# output layer 1 neuron = regression prediction
my_model.add(Dense(1))

# summary
print(my_model.summary())

# -----------------
# Initializing the optimizer and compiling the model
# -----------------
opt = Adam(learning_rate = 0.01)
my_model.compile(loss = 'mse', metrics = ['mae'], optimizer = opt)

# -----------------
# Fit and evaluate the model
# -----------------
my_model.fit(features_train_scaled, labels_train, epochs = 40, batch_size = 5, verbose = 0)
final_mse, final_mae = my_model.evaluate(features_test_scaled, labels_test, verbose = 0)
print('final MSE = ', final_mse)
print('final MAE = ', final_mae)