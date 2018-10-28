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

from mongo_engine import MongoEngine
import data_deal as dd
from histogram import Histogram


# build function for the Keras' scikit-learn API
def create_keras_model():
    """
    This function compiles and returns a Keras model.
    Should be passed to KerasClassifier in the Keras scikit-learn API.
    """

    model = Sequential()
    model.add(BatchNormalization(momentum=0.8, input_shape=(cf.ROW_LENGTH, 4,1)))
    model.add(Conv2D(32, kernel_size=(4, 2), activation='relu'))
    # model.add(Conv2D(64, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # model.add(BatchNormalization(momentum=0.8))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    return model


# create the classifier
classifier = KerasClassifier(create_keras_model)

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
# y_test = keras.utils.to_categorical(y_test, 10)

# assemble initial data
n_initial = 1000
idx = []
up_and_down_count = 0
not_order_count = 0
for i in range(len(y_list)):
    if y_list[i] == 1 or y_list[i] == -1 or y_list[i] == 2:
        up_and_down_count +=1
        idx.append(i)
y_00 = []
for i in range(len(y_list)):
    if y_list[i] == 0:
        y_00.append(i)
if len(y_00)<=up_and_down_count:
    idx = idx + y_00
else:
    y_0 = np.array(y_00)
    y_0_pos = np.random.choice(range(len(y_0)), size=up_and_down_count, replace=False)
    y_00 = y_0[y_0_pos].tolist()
    idx = idx + y_00

# initial_idx = np.random.choice(range(len(x_list)), size=n_initial, replace=False)
# idx.append(233)
# idx.append(323)
initial_idx = np.array(idx)
x_list = x_list.reshape(-1,cf.ROW_LENGTH,4,1)
X_initial = x_list[initial_idx]
# X_initial = X_initial.reshape(-1,cf.ROW_LENGTH,4,1)

y_initial = y_list[initial_idx]
y_initial = keras.utils.to_categorical(y_initial, 4)
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
def regression(regressor, X):
    x_up = []
    x_down = []
    x_damped = []
    pred = regressor.predict(X)
    for i in range(len(pred)):
        if pred[i] == 1:
            x_up.append(i)
        if pred[i] == 3:
            x_down.append(i)
        if pred[i] == 2:
            x_damped.append(i)
    x_up = np.array(x_up)
    x_down = np.array(x_down)
    x_damped = np.array(x_damped)
    index_up = cf.MANUAL_SIZE
    if len(x_up)>cf.MANUAL_SIZE:
        index_up = np.random.choice(range(len(x_up)), size=cf.MANUAL_SIZE, replace=False)
    index_down = cf.MANUAL_SIZE
    if len(x_down)>cf.MANUAL_SIZE:
        index_down = np.random.choice(range(len(x_down)), size=cf.MANUAL_SIZE, replace=False)
    index_damped = cf.MANUAL_SIZE
    if len(x_damped)>cf.MANUAL_SIZE:
        index_damped = np.random.choice(range(len(x_damped)), size=cf.MANUAL_SIZE, replace=False)
    xx = np.concatenate((x_up[index_up] if len(x_up)>cf.MANUAL_SIZE else x_up,x_down[index_down] if len(x_down)>cf.MANUAL_SIZE else x_down,x_damped[index_damped] if len(x_damped)>cf.MANUAL_SIZE else x_damped)).astype('int64')




    return xx, X[xx]



# initialize ActiveLearner
learner = ActiveLearner(
    estimator=classifier,
    query_strategy=regression,
    X_training=X_initial, y_training=y_initial,
    verbose=0
)


def plt_manual(xx):
    print(len(xx))
    xxx = []
    for i in range(len(xx)):
        manual_data = (x_pool[xx[i]],y_pool[xx[i]],t_pool[xx[i]],i_pool[xx[i]])
        histogram = Histogram(manual_data)
        histogram.createImage()
        y_pool[xx[i]] = histogram.result
        print(y_pool[xx[i]])
        if y_pool[xx[i]] != cf.NONE_SENSE_DATA_TYPE:
            xxx.append(xx[i])
        print(y_pool[xx[i]])

    return np.array(xxx)
learner.teach(
    X=X_initial, y=y_initial
)
# the active learning loop
n_queries = 10
for idx in range(n_queries):
    query_idx, query_instance = learner.query(x_pool)
    xx = plt_manual(query_idx)

    print(query_idx)
    learner.teach(
        X=x_pool[xx], y=keras.utils.to_categorical(y_pool[xx], 4)
    )
    # remove queried instance from pool
    X_pool = np.delete(x_pool, xx, axis=0)
    y_pool = np.delete(y_pool, xx, axis=0)
    t_pool = np.delete(t_pool, xx, axis=0)
    i_pool = np.delete(i_pool, xx, axis=0)
# the final accuracy score
# print(learner.score(X_test, y_test, verbose=1))
