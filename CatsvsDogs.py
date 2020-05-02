import os
abspath = os.path.abspath("C:/Users/miqui/OneDrive/Python Projects/Images/dogs-vs-cats")
os.chdir(abspath)
os.listdir()
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout, Dense, GlobalMaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model


# Preparing the Training Data

filenames = os.listdir("train")
categories = []
for filename in filenames:
    category = filename.split(".")[0]
    if category == "dog":
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
        "filename": filenames,
        "category": categories})

df.head(10)
df.columns

df["category"].value_counts().plot.bar()
plt.show()

# Building a Model
SIZE = 200 # Change accordingly
input_shape = (SIZE, SIZE, 3) # Set to 3 for colored images
epochs = 5
batch_size = 32

pretrained_model = VGG16(weights = "imagenet",
                         input_shape = input_shape,
                         include_top = False) #whether to include the fully-connected layer at the top

for layer in pretrained_model.layers[:15]:
    layer.trainable = False

for layer in pretrained_model.layers[15:]:
    layer.trainable = True

last_layer = pretrained_model.get_layer("block5_pool")
last_output = last_layer.output


x = GlobalMaxPooling2D()(last_output)
# Add a fully connected layer with 512 hidden units and ReLU activation
x = Dense(512, activation='relu')(x)
# Add a dropout rate of 0.5
x = Dropout(0.5)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pretrained_model.input, x)

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

model.summary()


train_df, validate_df = train_test_split(df, test_size = 0.1)
train_df = train_df.reset_index()
validate_df = validate_df.reset_index()

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_df["category"] = df["category"].astype(str)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    "train",
    x_col='filename',
    y_col='category',
    class_mode='binary',
    target_size=(SIZE, SIZE),
    batch_size=batch_size)

validate_df["category"] = validate_df["category"].astype(str)
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    "train",
    x_col='filename',
    y_col='category',
    class_mode='binary',
    target_size=(SIZE, SIZE),
    batch_size=batch_size)

history = model.fit_generator(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=total_validate//batch_size,
            steps_per_epoch=total_train//batch_size)























