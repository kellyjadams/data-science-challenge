# Random forest model
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

train_data = pd.read_csv("train.csv", header=None)
test_data = pd.read_csv("test.csv", header=None)

X_train = train_data.iloc[:, 2:5]
y_train = train_data.iloc[:, 5]
X_test = test_data.iloc[:, 2:5]

# Instantiate the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict probabilities for the test dataset
y_pred = model.predict_proba(X_test)[:, 1]

# Save the predicted probabilities to a text file
np.savetxt("labels_random_forest.txt", y_pred, fmt="%.7f")
