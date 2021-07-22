"""import tensorflow as tf
from tensorflow.keras import datasets, layers, models
model = models.Sequential()"""
"""import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('./', one_hot=True)
print(mnist)"""

test_data_file = open(r"C:\Users\DELL\Desktop\PMI\python\lpthw\神经网络训练数据\mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close  