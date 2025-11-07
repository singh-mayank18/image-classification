import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

#Load and process the dataset
data_dir = 'my_dataset'
filepaths = []
labels = []


for label in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, label)
    for file in os.listdir(folder_path):
        filepaths.append(os.path.join(folder_path, file))
        labels.append(label)

#Create a DataFrame
df = pd.DataFrame({'filepaths': filepaths, 'labels': labels})


#Split the dataset
train_df, temp_df = train_test_split(df, train_size=0.6, stratify=df['labels'], random_state=42)

valid_df, test_df = train_test_split(temp_df, train_size=0.5, stratify=temp_df['labels'], random_state=42)



#Step 2: Create Image Data Generators
img_size = (224, 224)
batch_size = 32

train_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', 
                                                                   target_size=img_size, class_mode='categorical', 
                                                                   batch_size=batch_size)

valid_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', 
                                                                   target_size=img_size, class_mode='categorical', 
                                                                   batch_size=batch_size)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', 
                                                                  target_size=img_size, class_mode='categorical', 
                                                                  batch_size=batch_size, shuffle=False)

#Build the CNN model
model = Sequential([
    tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(224, 224, 3), pooling='max'),
    BatchNormalization(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(train_gen.class_indices), activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

#Train the model
history = model.fit(train_gen, validation_data=valid_gen, epochs=10)

#Evaluate the model
test_loss, test_accuracy = model.evaluate(test_gen)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

#Save the model
model.save('image_classification_model.h5')

#Visualize training results
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
