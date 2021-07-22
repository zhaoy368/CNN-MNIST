import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy

class CNN(object):
    def __init__(self):
        model = models.Sequential()
        # 第1层卷积，卷积核大小为3*3，32个，28*28为待训练图片的大小
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        # 第2层卷积，卷积核大小为3*3，64个
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        # 第3层卷积，卷积核大小为3*3，64个
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        model.summary()

        self.model = model

class DataSource(object):
    def __init__(self):
        with open(r"C:\Users\DELL\Desktop\PMI\python\python\神经网络训练数据\mnist_train.csv") as data_file:
            data_list = data_file.readlines()
        with open(r"C:\Users\DELL\Desktop\PMI\python\lpthw\神经网络训练数据\mnist_test.csv") as test_data_file:
            test_data_list = test_data_file.readlines()
        train_images = []
        train_labels = []
        for record in data_list:
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)
            targets = int(all_values[0])
            #targets = numpy.zeros(10)
            #targets[int(all_values[0])] = 1
            inputs = inputs.reshape((28,28))
            train_images.append(inputs)
            train_labels.append(targets)
        train_images = numpy.array(train_images)
        train_labels = numpy.array(train_labels)       

        test_images = []
        test_labels = []
        for record in test_data_list:
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)
            targets = int(all_values[0])
            #targets = numpy.zeros(10)
            #targets[int(all_values[0])] = 1
            inputs = inputs.reshape((28,28))
            test_images.append(inputs)
            test_labels.append(targets)
        test_images = numpy.array(test_images)
        test_labels = numpy.array(test_labels)
        # mnist数据集存储的位置，如何不存在将自动下载
        #data_path = os.path.abspath(os.path.dirname(__file__)) + '/../data_set_tf2/mnist.npz'
        #(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data(path=data_path)
        # 6万张训练图片，1万张测试图片
        train_images = train_images.reshape((60000, 28, 28, 1))
        test_images = test_images.reshape((10000, 28, 28, 1))
        # 像素值映射到 0 - 1 之间
        train_images, test_images = train_images / 255.0, test_images / 255.0

        self.train_images, self.train_labels = train_images, train_labels
        self.test_images, self.test_labels = test_images, test_labels

class Train:
    def __init__(self):
        self.cnn = CNN()
        self.data = DataSource()

    def train(self):
        #check_path = './ckpt/cp-{epoch:04d}.ckpt'
        # period 每隔5epoch保存一次
        #save_model_cb = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True, verbose=1, period=5)

        self.cnn.model.compile(optimizer='adam',
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])
        self.cnn.model.fit(self.data.train_images, self.data.train_labels, batch_size=128, epochs=1)#, callbacks=[save_model_cb])

        test_loss, test_acc = self.cnn.model.evaluate(self.data.test_images, self.data.test_labels)
        print("准确率: %.4f，共测试了%d张图片 " % (test_acc, len(self.data.test_labels)))


if __name__ == "__main__":
    app = Train()
    app.train()