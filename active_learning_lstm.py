"""
This example demonstrates how to use the active learning interface with Keras.
The example uses the scikit-learn wrappers of Keras. For more info, see https://keras.io/scikit-learn-api/
"""

import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from modAL.models import ActiveLearner
import config as cf
from keras.layers import LSTM
from keras.layers import SimpleRNN
from mongo_engine import MongoEngine
import data_deal as dd
from histogram import Histogram
from keras.models import load_model
import os

# build function for the Keras' scikit-learn API
def create_keras_model():
    """
    This function compiles and returns a Keras model.
    Should be passed to KerasClassifier in the Keras scikit-learn API.
    """

    model = Sequential()
    model.add(BatchNormalization(momentum=0.8, input_shape=(cf.ROW_LENGTH, 4),axis=1))
    # model.add(LSTM(32,return_sequences=True))
    model.add(LSTM(32,return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    if os.path.exists(cf.LSTM_MODEL_NAME):
        model = load_model(cf.LSTM_MODEL_NAME)

    return model


# create the classifier
# classifier = KerasClassifier(create_keras_model)
classifier = create_keras_model()

"""
Data wrangling
1. Reading data from Keras
2. Assembling initial training data for ActiveLearner
3. Generating the pool
"""

# read training data
mongo_engine = MongoEngine()
list = mongo_engine.getAllData()
x_l,y_l,t_l,i_l = dd.list_data2train_data(list)
x_list = np.array(x_l)
y_list = np.array(y_l)
t_list = np.array(t_l)
i_list = np.array(i_l)

# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train = X_train.reshape(60000, 28, 28, 1).astype('float32') / 255
# X_test = X_test.reshape(10000, 28, 28, 1).astype('float32') / 255
# y_train = keras.utils.to_categorical(y_train, 10)
# y_test = keras.utils.to_catego rical(y_test, 10)

# assemble initial data
n_initial = 1000
idx = []
up_and_down_count = 0
not_order_count = 0
for i in range(len(y_list)):
    if y_list[i] == 1 or y_list[i] == -1 or y_list[i] == 2:
        up_and_down_count +=1
        idx.append(i)
order_idx = idx
not_order_idx = []
y_00 = []
for i in range(len(y_list)):
    if y_list[i] == cf.NOT_ORDER_DATA_TYPE:
        y_00.append(i)
# if len(y_00)<=up_and_down_count*cf.ORDER_TO_NOT_ORDER:
#     not_order_idx = y_00
# else:
#     y_0 = np.array(y_00)
#     y_0_pos = np.random.choice(range(len(y_0)), size=up_and_down_count*cf.ORDER_TO_NOT_ORDER, replace=False)
#     not_order_idx = y_0[y_0_pos].tolist()
not_order_idx = y_00
idx = order_idx + not_order_idx

# initial_idx = np.random.choice(range(len(x_list)), size=n_initial, replace=False)
# idx.append(233)
# idx.append(323)
initial_idx = np.array(idx)
# x_list = x_list.reshape(-1,cf.ROW_LENGTH,4,1)
X_initial = x_list[initial_idx]
# X_initial = X_initial.reshape(-1,cf.ROW_LENGTH,4,1)
# y_list = keras.utils.to_categorical(y_list, 4)
y_initial = y_list[initial_idx]
y_initial = keras.utils.to_categorical(y_initial, 4)


x_order = x_list[order_idx]
y_order = y_list[order_idx]
y_order = keras.utils.to_categorical(y_order, 4)

x_not_order = x_list[not_order_idx]
y_not_order = y_list[not_order_idx]
y_not_order = keras.utils.to_categorical(y_not_order, 4)


# y_list = keras.utils.to_categorical(y_list, 4)
t_initial = t_list[initial_idx]
i_initial = i_list[initial_idx]

# generate the pool
# remove the initial data from the training dataset
x_pool = np.delete(x_list, initial_idx, axis=0)
y_pool = np.delete(y_list, initial_idx, axis=0)
t_pool = np.delete(t_list, initial_idx, axis=0)
i_pool = np.delete(i_list, initial_idx, axis=0)

"""
Training the ActiveLearner
"""
# query strategy for regression
def sort_max(pred,size):
    yuan = ()
    for m in range(pred.shape[1]):
        list = []
        for i,e in enumerate(pred):
            insert_flag = False
            if len(list) == 0:
                list.append([i,e[m]])
                insert_flag = True
                continue
            for j in range(len(list)):
                if list[j][1]<e[m]:
                    list.insert(j,[i,e[m]])
                    insert_flag = True
                    break
            if len(list) < size and insert_flag == False:
                list.append([i,e[m]])
            if len(list) == size + 1:
                list.pop(size)
        p_list = []
        for o in list:
            p_list.append(o[0])
        np_list = np.array(p_list)
        yuan = yuan + (np_list,)
    return yuan


def regression(regressor, X):
    pred = regressor.predict(X)
    x_not_order,x_up,x_damped,x_down = sort_max(pred,cf.CHECK_IMAGE_GROUP_SIZE)
    xx = np.concatenate((x_not_order,x_up,x_damped,x_down))
    return xx, X[xx]



# initialize ActiveLearner
learner = ActiveLearner(
    estimator=classifier,
    query_strategy=regression,
    X_training=X_initial[:,::-1,:], y_training=y_initial,
    verbose=0
)


def plt_manual(xx):
    print(len(xx))
    xxx = []
    xxxx = []
    for i in range(len(xx)):
        manual_data = (x_pool[xx[i]],y_pool[xx[i]],t_pool[xx[i]],i_pool[xx[i]])
        histogram = Histogram(manual_data)
        histogram.createImage()
        y_pool[xx[i]] = histogram.result
        print(y_pool[xx[i]])
        if y_pool[xx[i]] != cf.NONE_SENSE_DATA_TYPE:
            xxx.append(xx[i])
        if y_pool[xx[i]] != cf.NONE_SENSE_DATA_TYPE and y_pool[xx[i]] != cf.NOT_ORDER_DATA_TYPE:
            xxxx.append(xx[i])

    return np.array(xxx),np.array(xxxx)
# learner.teach(
#     X=X_initial, y=y_initial
# )
learner.teach(
    X=x_not_order[:,::-1,:], y=y_not_order
)
for i in range(cf.ORDER_TO_NOT_ORDER):
    learner.teach(
        X=x_order[:,::-1,:], y=y_order
    )

# the active learning loop
n_queries = 10
for idx in range(n_queries):
    query_idx, query_instance = learner.query(x_pool[:,::-1,:])
    xx,xxxx = plt_manual(query_idx)

    print(query_idx)
    if len(xx) != 0:
        learner.teach(
            X=x_pool[xx][:,::-1,:], y=keras.utils.to_categorical(y_pool[xx],4)
        )
    if len(xxxx) != 0:
        for i in range(cf.ORDER_TO_NOT_ORDER):
            learner.teach(
                X=x_pool[xxxx][:,::-1,:], y=keras.utils.to_categorical(y_pool[xxxx],4)
            )
    # remove queried instance from pool
    x_pool = np.delete(x_pool, xx, axis=0)
    y_pool = np.delete(y_pool, xx, axis=0)
    t_pool = np.delete(t_pool, xx, axis=0)
    i_pool = np.delete(i_pool, xx, axis=0)
    print(len(x_pool))
    print(len(y_pool))
    print(len(t_pool))
    print(len(i_pool))

# the final accuracy score
# print(learner.score(X_test, y_test, verbose=1))
classifier.save(cf.LSTM_MODEL_NAME)