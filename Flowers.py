import cv2
import os
abspath = os.path.abspath("C:/Users/miqui/OneDrive/Python Projects/Images/Flowers")
os.chdir(abspath)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop


# 5 Classes
# Each class has about 800 photos

# Creating directories:
daisy = os.path.join("daisy")
dandelion = os.path.join("dandelion")
rose = os.path.join("rose")
sunflower = os.path.join("sunflower")
tulip = os.path.join("tulip")

# Test to make sure they are in our directory:
os.listdir(tulip)

print('total daisy images:', len(os.listdir(daisy)))
print('total dandelion images:', len(os.listdir(dandelion)))
print('total rose images:', len(os.listdir(rose)))
print('total sunflower images:', len(os.listdir(sunflower)))
print('total tulip images:', len(os.listdir(tulip)))

# Rescale images:

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
                    "C:/Users/miqui/OneDrive/Python Projects/Images/Flowers",
                     target_size=(200, 200),
                     batch_size = batch_size,
                     color_mode="rgb",
                     classes=["daisy", "dandelion", "rose", "sunflower", "tulip"],
                     class_mode="categorical")

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 200x 200 with 3 bytes color
    # The first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a dense layer
    tf.keras.layers.Flatten(),
    # 128 neuron in the fully-connected layer
    tf.keras.layers.Dense(128, activation='relu'),
    # 5 output neurons for 5 classes with the softmax activation
    tf.keras.layers.Dense(5, activation='softmax')
])

model.summary()

# Compiling the model:
model.compile(optimizer=RMSprop(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])


batch_size = 128
epochs = 20
nsamples = train_generator.n

history = model.fit_generator(
            train_generator,
            steps_per_epoch=int(nsamples/batch_size),
            shuffle=True,
            epochs=epochs,
            verbose=1)

