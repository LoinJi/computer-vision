# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 13:13:11 2019

@author: 咸鸡
"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

from keras import models
from keras import layers
from keras.optimizers import RMSprop



categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']

newsgroups_train = fetch_20newsgroups(subset='train',  categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',  categories=categories)

num_train = len(newsgroups_train.data)#newsgroups_train.data就是一封邮件或是新闻的文本内容
num_test  = len(newsgroups_test.data)#测试集

# max_features是一个重要的参数，你需要调整它.
vectorizer = TfidfVectorizer(max_features=10000) 
#TfidfVectorizer可以把原始文本转化为tf-idf的特征矩阵，从而为后续的文本相似度计算

X = vectorizer.fit_transform( newsgroups_train.data + newsgroups_test.data )
#就是求得训练集X的均值，方差，最大值，最小值,这些训练集X固有的属性
#在fit的基础上，进行标准化，降维，归一化等操作
#fit_transform是fit和transform的组合，既包括了训练又包含了转换

X_train = X[0:num_train, :]
X_test = X[num_train:num_train+num_test,:]

all_data=X_train.toarray() 
all_data=list(all_data)
all_texts=[]
for i in range(0,len(all_data)):
    all_texts.append(list(all_data[i]))

all_data1=X_test.toarray() 
all_data1=list(all_data1)
all_texts1=[]
for i in range(0,len(all_data1)):
    all_texts1.append(list(all_data1[i]))


Y_train = newsgroups_train.target
Y_test = newsgroups_test.target

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

def to_one_hot(labels, dimension=4):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels): #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
        results[i, label] = 1
    return results

one_hot_train_labels = to_one_hot(Y_train)
one_hot_test_labels = to_one_hot(Y_test)

#train_texts_array=np.array(train_texts)
#test_texts_array=np.array(test_texts)

model = models.Sequential()
model.add(layers.Dense(units=8, activation='relu', input_shape=(10000, )))
#activation为激活函数，该层为输入层
model.add(layers.Dense(units=8, activation='relu'))#该层为隐层

model.add(layers.Dense(units=4, activation='softmax'))  # 一共4类，该层为输出层

model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])#编译模型

train_data_list=np.array(all_texts)
test_data_list=np.array(all_texts1)

#  开始训练网络

model.fit(train_data_list, one_hot_train_labels, epochs=5, batch_size=16, validation_data=[test_data_list, one_hot_test_labels])

# 来在测试集上测试一下模型的性能吧

test_loss, test_accuracy = model.evaluate(test_data_list, one_hot_test_labels)
print("test_loss:", test_loss, "    test_accuracy:", test_accuracy)

x_test1=np.array(all_texts1)
predict_test=model.predict_classes(x_test1).astype('int')
print(predict_test)

count=0
predict_test=list(predict_test)
Y_test1=list(Y_test)
for i in range(0,len(predict_test)):
    if predict_test[i]==Y_test1[i]:
        count=count+1
print(count/len(Y_test1))

'''
#以下就是线性模型分类的过程
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X_train, Y_train) #训练集训练

Y_predict = clf.predict(X_test) #用训练得到的模型测试
'''


'''
print(Y_test)
print(Y_predict)

ncorrect = 0
for dy in  (Y_test - Y_predict):
	if 0 == dy:
		ncorrect += 1

print('text classification accuracy is {}%'.format(round(100.0*ncorrect/len(Y_test)) ) )
'''