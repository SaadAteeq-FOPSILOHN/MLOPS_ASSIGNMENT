import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Load data [cite: 39]
data = pd.read_csv('data/dataset.csv')
X = data[['feature1', 'feature2']]
y = data['target']

# Train model [cite: 40]
model = LogisticRegression()
model.fit(X, y)

# Save model [cite: 41]
os.makedirs('models', exist_ok=True)
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Training completed and model saved to models/model.pkl")