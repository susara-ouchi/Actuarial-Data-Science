import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load your dataset
# Assume df is a pandas DataFrame with your data
# df = pd.read_csv('your_life_insurance_dataset.csv')

# Example data (replace this with your actual dataset)
data = {
    'age': [30, 60, 45, 50, 25],
    'gender': [1, 0, 1, 0, 1],  # 1: Male, 0: Female
    'smoker': [0, 1, 0, 1, 0],  # 1: Smoker, 0: Non-smoker
    'bmi': [22.5, 28.7, 24.3, 30.1, 20.2],
    'coverage_amount': [100000, 200000, 150000, 250000, 120000],
    'claims': [0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# Features and target
X = df.drop('claims', axis=1)
y = df['claims']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=10, callbacks=[early_stopping])

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test Accuracy: {accuracy:.2f}')

# Predict the likelihood of claims for new data
new_data = np.array([[40, 1, 0, 25.4, 180000]])  # Example new data
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print(f'Predicted likelihood of claim: {prediction[0][0]:.2f}')
