import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Paths to your CSV files and image directories
train_csv_path = 'training_ground_truth.csv'
train_image_dir = 'original_training_images'
test_csv_path = 'test_ground_truth.csv'
test_image_dir = 'Test_Data'

# Load CSV files
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# Print the columns to check names
print("Training DataFrame columns:", train_df.columns)
print("Test DataFrame columns:", test_df.columns)

# Rename the columns to 'Image' and 'Label' for training data
train_df.columns = ['Image', 'Label']

# Rename the columns to 'Image' and 'Label' for testing data
test_df.columns = ['Image', 'Label']

# Add .jpg extension to the filenames
train_df['Image'] = train_df['Image'] + '.jpg'
test_df['Image'] = test_df['Image'] + '.jpg'

# Convert test labels to strings
test_df['Label'] = test_df['Label'].astype(str)

# Preview the data
print(train_df.head())
print(test_df.head())

# List a few files from the image directories
print("Sample training images:", os.listdir(train_image_dir)[:5])
print("Sample test images:", os.listdir(test_image_dir)[:5])

# Check if images exist in the directory
train_images = set(os.listdir(train_image_dir))
test_images = set(os.listdir(test_image_dir))

# Filter out invalid filenames
train_df = train_df[train_df['Image'].isin(train_images)]
test_df = test_df[test_df['Image'].isin(test_images)]

# Debug: Print the number of valid filenames found
print(f"Number of valid training images: {len(train_df)}")
print(f"Number of valid test images: {len(test_df)}")

# Ensure there are at least two classes in the training data
print(f"Unique labels in training data: {train_df['Label'].unique()}")

# Image data generator for training
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2  # Use 20% of training data for validation
)

# Image data generator for test
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Create training and validation generators
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=train_image_dir,
    x_col='Image',
    y_col='Label',
    target_size=(299, 299),  # Xception input size
    batch_size=32,
    class_mode='binary',  # Change to 'categorical' if you have more than 2 classes
    subset='training'
)

validation_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=train_image_dir,
    x_col='Image',
    y_col='Label',
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Test generator (with labels)
test_generator = test_datagen.flow_from_dataframe(
    test_df,
    directory=test_image_dir,
    x_col='Image',
    y_col='Label',
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Load the Xception model pre-trained on ImageNet, excluding the top layers
base_model = Xception(weights='imagenet', include_top=False)

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  # Change to the number of classes for multiclass classification

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
history = model.fit(
    train_generator,
    epochs=10,  # Adjust the number of epochs as needed
    validation_data=validation_generator
)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator), verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Predict the labels for test data
predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)

# Convert predictions to class labels
predicted_classes = np.where(predictions > 0.5, 1, 0)

# Add predictions to the test dataframe
test_df['Predicted_Labels'] = predicted_classes

# Save the results
test_df.to_csv('predicted_labels.csv', index=False)
