import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load the data
train_data = pd.read_csv("train.csv", header=None)
test_data = pd.read_csv("test.csv", header=None)

# Extract input features (X) and output labels (y) for training
X_train = train_data.iloc[:, 2:5]
y_train = train_data.iloc[:, 5]
X_test = test_data.iloc[:, 2:5]

# Create and train the logistic regression model
model = LogisticRegression(solver="liblinear")
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict_proba(X_test)[:, 1]

# Save the predictions to labels.txt
np.savetxt("labels_logistic.txt", y_pred, fmt="%.7f")
