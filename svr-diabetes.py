import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Load the diabetes dataset
dataset = datasets.load_diabetes()

# Convert to DataFrame and add target column
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target

# Separate features and target
X = df.drop(columns='target')
y = df['target']

# Split the data into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# Standardize the features (important for SVMs)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the list of kernels to test
kernels = ["linear", "poly", "rbf", "sigmoid"]

# Train and evaluate a model for each kernel
for ker in kernels:
    model = SVR(kernel=ker)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'Model with {ker} kernel error: {rmse:.3f}')
