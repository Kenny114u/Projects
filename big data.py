#!/usr/bin/env python
# coding: utf-8

# In[23]:


import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np



parent_directory = "/Users/mac/Downloads/datab/dataset/real-vs-fake/train/"

subdirectories = ["fake", "real"]
for subdir in subdirectories:
    subdir_path = os.path.join(parent_directory, subdir)
    if os.path.isdir(subdir_path):
        print(f"Displaying images from {subdir} directory:")
        image_files = [f for f in os.listdir(subdir_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
        num_images_to_display = min(len(image_files), 5)
        for i in range(num_images_to_display):
            image_path = os.path.join(subdir_path, image_files[i])
            
            img = Image.open(image_path)
            plt.imshow(img)
            plt.title(image_files[i])
            plt.axis('off')
            plt.show()
    else:
        print(f"Subdirectory '{subdir}' not found.")





# In[13]:


class_counts = {}

for subdir in subdirectories:
    subdir_path = os.path.join(parent_directory, subdir)
    if os.path.isdir(subdir_path):

        image_files = [f for f in os.listdir(subdir_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        class_counts[subdir] = len(image_files)

plt.figure(figsize=(8, 6))
plt.bar(class_counts.keys(), class_counts.values())
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.show()


# In[14]:


widths = []
heights = []

for subdir in subdirectories:
    subdir_path = os.path.join(parent_directory, subdir)
    if os.path.isdir(subdir_path):
        image_files = [f for f in os.listdir(subdir_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        for image_file in image_files:
            image_path = os.path.join(subdir_path, image_file)
            img = Image.open(image_path)
            width, height = img.size
            widths.append(width)
            heights.append(height)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(widths, bins=30, color='skyblue', alpha=0.7)
plt.title('Image Width Distribution')
plt.xlabel('Width')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(heights, bins=30, color='salmon', alpha=0.7)
plt.title('Image Height Distribution')
plt.xlabel('Height')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()



# In[3]:


import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

parent_directory = "/Users/mac/Downloads/data/dataset/real-vs-fake/image/"
subdirectories = ["fake", "real"]


num_images_per_channel = 50  

red_values = []
green_values = []
blue_values = []

target_resolution = (28, 28)

for subdir in subdirectories:
    subdir_path = os.path.join(parent_directory, subdir)
    if os.path.isdir(subdir_path):
        image_files = [f for f in os.listdir(subdir_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

        
        np.random.shuffle(image_files)

        selected_image_files = image_files[:num_images_per_channel]

        for image_file in selected_image_files:
            image_path = os.path.join(subdir_path, image_file)
            img = Image.open(image_path).convert("RGB")
            img = img.resize(target_resolution)
            img_array = np.array(img)

            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                red_values.extend(img_array[:, :, 0].flatten())
                green_values.extend(img_array[:, :, 1].flatten())
                blue_values.extend(img_array[:, :, 2].flatten())
            else:
                print(f"Ignoring {image_file}: Not an RGB image")
    else:
        print(f"Subdirectory '{subdir}' not found.")

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.hist(red_values, bins=30, color='red', alpha=0.7)
plt.title('Red Channel Distribution')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.hist(green_values, bins=30, color='green', alpha=0.7)
plt.title('Green Channel Distribution')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
plt.hist(blue_values, bins=30, color='blue', alpha=0.7)
plt.title('Blue Channel Distribution')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.show()


# In[4]:


import os
import shutil
import random



test_split = 0.2

train_directory = os.path.join(parent_directory, "train")
test_directory = os.path.join(parent_directory, "test")
os.makedirs(train_directory, exist_ok=True)
os.makedirs(test_directory, exist_ok=True)

for subdir in subdirectories:
    subdir_path = os.path.join(parent_directory, subdir)
    if os.path.isdir(subdir_path):
        image_files = [f for f in os.listdir(subdir_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(image_files)
        num_test_images = int(len(image_files) * test_split)
        for i in range(num_test_images):
            src = os.path.join(subdir_path, image_files[i])
            dst = os.path.join(test_directory, subdir, image_files[i])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
        for i in range(num_test_images, len(image_files)):
            src = os.path.join(subdir_path, image_files[i])
            dst = os.path.join(train_directory, subdir, image_files[i])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)

print("Dataset split into train and test directories successfully.")


# In[4]:


import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

train_dir = "/Users/mac/Downloads/data/dataset/real-vs-fake/image/train"
test_dir = "/Users/mac/Downloads/data/dataset/real-vs-fake/image/test"

image_size = 212
img_rows, img_cols = image_size, image_size
num_classes = 2

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=100,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(image_size, image_size),
    batch_size=40,
    class_mode='categorical'
)


model = Sequential([
    Conv2D(30, kernel_size=(5, 5), strides=2, activation='relu', input_shape=(img_rows, img_cols, 3)),
    Conv2D(60, kernel_size=(5, 5), strides=2, activation='relu'),
    Conv2D(120, kernel_size=(5, 5), activation='relu'),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
 steps_per_epoch=1270,
    epochs=1,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)

test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print("\nTest accuracy:", test_acc)


# In[1]:


import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

train_dir = "/Users/mac/Downloads/data/dataset/real-vs-fake/image/train"
test_dir = "/Users/mac/Downloads/data/dataset/real-vs-fake/image/test"

image_size = 212
img_rows, img_cols = image_size, image_size
num_classes = 2

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=100,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(image_size, image_size),
    batch_size=40,
    class_mode='categorical'
)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(256, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
 steps_per_epoch=1270,
    epochs=1,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)

test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print("\nTest accuracy:", test_acc)



# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

predictions = model.predict(test_generator)

class_labels = list(train_generator.class_indices.keys())

num_images_to_display = 20
for i in range(min(num_images_to_display, len(test_generator.filenames))):
    img_path = os.path.join(test_dir, test_generator.filenames[i])
    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')

    predicted_class = class_labels[np.argmax(predictions[i])]
    probability = np.max(predictions[i])

    plt.title(f'Predicted class: {predicted_class}, Probability: {probability:.4f}')
    plt.show()



# In[14]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Load the image
img_path =  "/Users/mac/Downloads/real.jpeg" # Provide the path to your custom image
img = image.load_img(img_path, target_size=(image_size, image_size))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.  # Rescale pixel values to [0, 1]

# Make prediction
prediction = model.predict(img_array)

# Get class labels
class_labels = list(train_generator.class_indices.keys())

# Display the image
plt.imshow(img)
plt.axis('off')

# Display the prediction
predicted_class = class_labels[np.argmax(prediction)]
probability = np.max(prediction)
plt.title(f'Predicted class: {predicted_class}, Probability: {probability:.4f}')
plt.show()


# In[5]:


import os
import numpy as np
from sklearn.metrics import classification_report

test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print("\nTest accuracy:", test_acc)

predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_generator.classes

report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print("\nClassification Report:")
print(report)


# In[12]:


import os

train_dir = "/Users/mac/Downloads/data/dataset/real-vs-fake/image/train"

subdirectories = [subdir for subdir in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, subdir))]

for subdir in subdirectories:
    subdir_path = os.path.join(train_dir, subdir)
    num_files = len(os.listdir(subdir_path))
    print(f"Number of items in {subdir}: {num_files}")


# In[13]:


import os

train_dir = "/Users/mac/Downloads/data/dataset/real-vs-fake/image/test"

subdirectories = [subdir for subdir in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, subdir))]

for subdir in subdirectories:
    subdir_path = os.path.join(train_dir, subdir)
    num_files = len(os.listdir(subdir_path))
    print(f"Number of items in {subdir}: {num_files}")


# In[15]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

img_path =  "/Users/mac/Downloads/fake.jpeg" 
img = image.load_img(img_path, target_size=(image_size, image_size))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.  

prediction = model.predict(img_array)

class_labels = list(train_generator.class_indices.keys())

plt.imshow(img)
plt.axis('off')

predicted_class = class_labels[np.argmax(prediction)]
probability = np.max(prediction)
plt.title(f'Predicted class: {predicted_class}, Probability: {probability:.4f}')
plt.show()


# In[16]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

img_path =  "/Users/mac/Downloads/jude.jpeg" 
img = image.load_img(img_path, target_size=(image_size, image_size))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.  

prediction = model.predict(img_array)

class_labels = list(train_generator.class_indices.keys())

plt.imshow(img)
plt.axis('off')

predicted_class = class_labels[np.argmax(prediction)]
probability = np.max(prediction)
plt.title(f'Predicted class: {predicted_class}, Probability: {probability:.4f}')
plt.show()


# In[19]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

img_path =  "/Users/mac/Downloads/foot.jpeg" 
img = image.load_img(img_path, target_size=(image_size, image_size))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.  ]

prediction = model.predict(img_array)

class_labels = list(train_generator.class_indices.keys())

plt.imshow(img)
plt.axis('off')

predicted_class = class_labels[np.argmax(prediction)]
probability = np.max(prediction)
plt.title(f'Predicted class: {predicted_class}, Probability: {probability:.4f}')
plt.show()


# In[ ]:




