import keras
from keras.integration_test.preprocessing_test_utils import preprocessing
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense

main_path = '/Users/ilabelozerov/Downloads/dataset-resized'
train_dir = '/Users/ilabelozerov/Downloads/dataset-resized/train'
valid_dir = '/Users/ilabelozerov/Downloads/dataset-resized/valid'
test_dir = '/Users/ilabelozerov/Downloads/dataset-resized/test'
width, height = 150, 150
input_shape = (width, height, 3)
epochs = 30
batch_size = 32

train_size = 1766
valid_size = 378
test_size = 383

datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='categorical')

valid_generator = datagen.flow_from_directory(
    valid_dir,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='categorical')

data_augmentation = keras.Sequential(
    [
        preprocessing.RandomFlip("horizontal"),
        preprocessing.RandomRotation(0.1),
        preprocessing.RandomZoom(0.1),
    ]
)
normalizer = preprocessing.Normalization()
# normalizer.adapt(train_generator)

inputs = keras.Input(shape=(32, 32, 3), name='img')
x = data_augmentation(inputs)
x = preprocessing.Rescaling(1.0 / 255)(x)
x = normalizer(inputs)
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.Conv2D(64, 3, activation='relu')(x)
block_1_output = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_1_output)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
block_2_output = layers.add([x, block_1_output])
x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_2_output)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
block_3_output = layers.add([x, block_2_output])
x = layers.Conv2D(64, 3, activation='relu')(block_3_output)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(6, activation='softmax')(x)

model = keras.Model(inputs, outputs, name='toy_resnet')

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# callbacks = ModelCheckpoint('models/weights.hdf5', monitor='val_accuracy', save_best_only=True, verbose=1)

model.fit_generator(
    train_generator,
    steps_per_epoch=train_size // batch_size,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=valid_size // batch_size)


# model.save('models/model')
# json_string = model.to_json()
# with open('models/model.json', 'w') as json_file:
#     json_file.write(json_string)



