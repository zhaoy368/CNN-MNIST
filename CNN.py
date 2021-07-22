import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
# 加载数据
mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()
# 数据处理
train_x0 = np.zeros((train_x.shape[0],32,32))
test_x0 = np.zeros((test_x.shape[0],32,32))
train_x0[:,2:30,2:30] = train_x
test_x0[:,2:30,2:30] = test_x
train_x = train_x0.reshape(-1,32,32,1)
test_x =  test_x0.reshape(-1,32,32,1)
X_train, X_test = tf.cast(train_x/255.0, tf.float32), tf.cast(test_x/255.0, tf.float32)
y_train, y_test = tf.cast(train_y, tf.int16), tf.cast(test_y, tf.int16)
# 模型创建
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(16, (5, 5), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(120, (4, 4), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(84, activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))
model.summary()
# 模型训练
model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy'])

history = model.fit(X_train, y_train, epochs=5, validation_split=0.2)
# 模型评价
model.evaluate(X_test, y_test, verbose=2)
# 可视化
#print(history.history.keys())
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']

plt.figure()
plt.subplot(121)
plt.plot(loss, color='blue', label='train')
plt.plot(val_loss, color="red",label='test')
plt.ylabel('Loss')
plt.legend()

plt.subplot(122)
plt.plot(acc, color='blue', label='train')
plt.plot(val_acc, color='red', label='test')
plt.ylabel('Accuracy')

plt.legend()
plt.show()