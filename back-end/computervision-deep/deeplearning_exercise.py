import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os

# 1. Load Data with Validation Split
# We use a validation split to monitor overfitting during training.
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128, 128),
    batch_size=8
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    batch_size=8
)

# Get class names dynamically from the dataset
class_names = train_ds.class_names
print(f"Classes found: {class_names}")

# Optimize performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 2. Data Augmentation
# Helps the model generalize better by creating variations of images.
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal", input_shape=(128, 128, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# 3. Model Architecture
model = keras.Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Dropout helps prevent overfitting
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 4. Training
epochs = 15
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# 5. Save the model
model.save('exercise_model.keras')
print("Model saved to exercise_model.keras")

# 6. Inference Example
# Only run this if the test image exists
test_image_path = "test_pushup.jpg"
if os.path.exists(test_image_path):
    img = cv2.imread(test_image_path)
    img = cv2.resize(img, (128, 128))
    # Note: No manual division by 255.0 here because the model has a Rescaling layer
    img_array = np.expand_dims(img, axis=0)

    prediction = model.predict(img_array)
    score = tf.nn.softmax(prediction[0])
    
    predicted_class = class_names[np.argmax(prediction)]
    confidence = 100 * np.max(prediction)

    print(f"Detected exercise: {predicted_class} with {confidence:.2f}% confidence")
else:
    print(f"Test image '{test_image_path}' not found. Skipping inference.")
