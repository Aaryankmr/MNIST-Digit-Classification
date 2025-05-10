import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load CSV data
train_df = pd.read_csv("mnist_train.csv")
test_df = pd.read_csv("mnist_test.csv")

# Separate features and labels
x_train = train_df.iloc[:, 1:].values / 255.0  # Normalize pixel values
y_train = to_categorical(train_df.iloc[:, 0].values, 10)  # One-hot encode labels

x_test = test_df.iloc[:, 1:].values / 255.0
y_test = to_categorical(test_df.iloc[:, 0].values, 10)

# Build the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"\n✅ Test accuracy: {accuracy:.4f}")

# OPTIONAL: Predict and visualize a few digits
def plot_sample(index):
    image = x_test[index].reshape(28, 28)
    plt.imshow(image, cmap='gray')
    pred = np.argmax(model.predict(x_test[[index]]))
    true = np.argmax(y_test[index])
    plt.title(f"Predicted: {pred}, Actual: {true}")
    plt.axis('off')
    plt.show()

# Show some predictions
for i in range(3):
    plot_sample(i)
