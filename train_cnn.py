import os
import numpy as np

from extract_spectrogram import extract_spectrogram

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split


real_folder = "dataset/real"
fake_folder = "dataset/fake"

X = []
y = []


print("Loading real voices...")

for file in os.listdir(real_folder):

    path = os.path.join(real_folder, file)

    spec = extract_spectrogram(path)

    X.append(spec)

    y.append(1)


print("Loading fake voices...")

for file in os.listdir(fake_folder):

    path = os.path.join(fake_folder, file)

    spec = extract_spectrogram(path)

    X.append(spec)

    y.append(0)


X = np.array(X)
y = np.array(y)


# reshape for CNN
X = X[..., np.newaxis]


# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)


# build CNN model
model = Sequential()

model.add(Conv2D(
    32,
    (3,3),
    activation='relu',
    input_shape=X.shape[1:]
))

model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


print("Training model...")

model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)


model.save("voice_model.h5")


print("MODEL TRAINED AND SAVED")