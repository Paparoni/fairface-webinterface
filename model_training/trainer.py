import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input

img_width = 64
img_height = 64

classes = {'race': ['Asian', 'Black', 'Indian', 'White', 'Hispanic'], 
           'sex': ['Female', 'Male'], 
           'age': ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']}

data = []
labels = []

for file in os.listdir('train_imgs'):
    race, sex, age, _ = file.split('_')
    image = cv2.imread(os.path.join('train_imgs', file))
    image = cv2.resize(image, (img_width, img_height))
    data.append(image)
    labels.append({'race': race, 'sex': sex, 'age': age})

data = np.array(data, dtype=np.float32) / 255.0
print(len(data))
inputs = Input(shape=(img_width, img_height, 3))

x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)

race_output = Dense(len(classes['race']), activation='softmax', name='race')(x)
sex_output = Dense(len(classes['sex']), activation='softmax', name='sex')(x)
age_output = Dense(len(classes['age']), activation='softmax', name='age')(x)

model = Model(inputs=inputs, outputs=[race_output, sex_output, age_output])

model.compile(optimizer='adam',
              loss={'race': 'categorical_crossentropy', 'sex': 'categorical_crossentropy', 'age': 'categorical_crossentropy'},
              metrics={'race': 'accuracy', 'sex': 'accuracy', 'age': 'accuracy'})

labels_dict = {key: np.array([classes[key].index(label[key]) for label in labels]) for key in classes}
print(labels_dict)
X_train, X_test, y_train, y_test = train_test_split(data, labels_dict, test_size=0.2, random_state=42)

model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=10, batch_size=32)

model.save('face_classification_model.h5')
