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
# import active_learning_rnn_close as alrc
def get_close_data(x_list):
    if cf.ROW_LENGTH != 128 and cf.ROW_LENGTH != 64 and cf.ROW_LENGTH !=32 and cf.ROW_LENGTH != 16:
        raise NameError
    if cf.ROW_LENGTH == 64:
        xx = np.delete(x_list,[0,1,2],axis=2)
        xxx = np.reshape(xx,[-1,cf.FIX_ROW_LENGTH])
        mean = np.mean(xxx,axis=1)
        mean = np.reshape(mean,[-1,1])
        xxxx = (xxx-mean)/mean
        xxxxx = np.reshape(xxxx,[-1,cf.FIX_ROW_LENGTH,1])
        return xxxxx
    if cf.ROW_LENGTH == 16:
        xxx = np.reshape(x_list,[-1,cf.FIX_ROW_LENGTH])
        mean = np.mean(xxx,axis=1)
        mean = np.reshape(mean,[-1,1])
        xxxx = (xxx-mean)/mean*4
        xxxxx = np.reshape(xxxx,[-1,cf.FIX_ROW_LENGTH,1])
        return xxxxx
    if cf.ROW_LENGTH == 32:
        xx = np.delete(x_list,[1,2],axis=2)
        xxx = np.reshape(xx,[-1,cf.FIX_ROW_LENGTH])
        mean = np.mean(xxx,axis=1)
        mean = np.reshape(mean,[-1,1])
        xxxx = (xxx-mean)/mean*2
        xxxxx = np.reshape(xxxx,[-1,cf.FIX_ROW_LENGTH,1])
        return xxxxx
    if cf.ROW_LENGTH == 128:
        xx = np.delete(x_list,[0,1,2],axis=2)
        xxx = np.reshape(xx,[-1,cf.ROW_LENGTH])
        xxx = np.reshape(xxx,[-1,cf.FIX_ROW_LENGTH,2])
        xxx = np.mean(xxx,axis=2)
        xxx = np.reshape(xxx,[-1,cf.FIX_ROW_LENGTH])
        mean = np.mean(xxx,axis=1)
        mean = np.reshape(mean,[-1,1])
        xxxx = (xxx-mean)/mean/2
        xxxxx = np.reshape(xxxx,[-1,cf.FIX_ROW_LENGTH,1])
        return xxxxx

# build function for the Keras' scikit-learn API
def create_keras_model():
    """
    This function compiles and returns a Keras model.
    Should be passed to KerasClassifier in the Keras scikit-learn API.
    """

    # model = Sequential()
    # model.add(BatchNormalization(momentum=0.8, input_shape=(cf.ROW_LENGTH, 4),axis=1))
    # # model.add(SimpleRNN(32,return_sequences=True))
    # model.add(SimpleRNN(32,return_sequences=True))
    # model.add(SimpleRNN(32))
    # model.add(Dense(4, activation='softmax'))
    #
    # model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    if os.path.exists(cf.GENERAL_MODEL_NAME):
        model = load_model(cf.GENERAL_MODEL_NAME)

        return model
    else:
        return None

mongo_engine = MongoEngine()
list = mongo_engine.getAllData()
x_l,y_l,t_l,i_l = dd.list_data2train_data(list)
x_list = np.array(x_l)
y_list = np.array(y_l)
t_list = np.array(t_l)
i_list = np.array(i_l)

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

model = create_keras_model()
x_close = get_close_data(x_pool)
pred = model.predict(x_close[:,::-1,:])
y_pool = np.argmax(pred,axis=1)
mongo_engine = MongoEngine()
righta = 0
wronga = 0
rightb = 0
wrongb = 0
ra = 0
rb = 0
wa = 0
wb = 0
total = 1

cal_size = int(cf.ROW_LENGTH/16)
wa123 = []
ga123 = []
for i in range(len(pred)):
    if i == 0:
        continue
    if y_pool[i] == 0:
        continue
    # if i_pool[i] == i_pool[i-cal_size]:
    #     total *= ((pred[i][1] - pred[i][3])*(x_pool[i-cal_size][0][3] - x_pool[i][0][3])/x_pool[i][0][3])+1
    if (y_pool[i] == 1) and i_pool[i] == i_pool[i-cal_size]:
        wa123.append(pred[i][1] - pred[i][3])
        ga123.append(x_pool[i-cal_size][0][3] - x_pool[i][0][3])
        # ha = (x_pool[i],pred[i],t_pool[i],i_pool[i])
        # histo = Histogram(ha)
        # histo.createImage()
        # da = mongo_engine.find_post_one(i_pool[i],t_pool[i])
        # if da is None:
        #     continue
        # total *= ((pred[i][1] - pred[i][3])*(x_pool[i-cal_size][0][3] - x_pool[i][0][3])/x_pool[i][0][3])+1
        # print(total)
        if x_pool[i][0][3] < x_pool[i-cal_size][0][3]:
            righta += 1
            ra += (x_pool[i-cal_size][0][3] - x_pool[i][0][3])/x_pool[i][0][3]
        else:
            wronga += 1
            wa += (x_pool[i-cal_size][0][3] - x_pool[i][0][3])/x_pool[i][0][3]
    elif y_pool[i] == 3 and i_pool[i] == i_pool[i-cal_size]:
        # ha = (x_pool[i],pred[i],t_pool[i],i_pool[i])
        # histo = Histogram(ha)
        # histo.createImage()
        # da = mongo_engine.find_post_one(i_pool[i],t_pool[i])
        # if da is None:
        #     continue
        total *= ((pred[i][3] - pred[i][1])*(-(x_pool[i-cal_size][0][3] - x_pool[i][0][3]))/x_pool[i][0][3])+1
        print(total)
        wa123.append(pred[i][1] - pred[i][3])
        ga123.append(x_pool[i-cal_size][0][3] - x_pool[i][0][3])
        if ((pred[i][3] - pred[i][1])*(-(x_pool[i-cal_size][0][3] - x_pool[i][0][3]))/x_pool[i][0][3])+1<0:
            exit(0)

        if x_pool[i][0][3] > x_pool[i-cal_size][0][3]:
            rightb += 1
            rb += (x_pool[i][0][3]-x_pool[i-cal_size][0][3])/x_pool[i][0][3]
        else:
            wrongb += 1
            wb += (x_pool[i][0][3]-x_pool[i-cal_size][0][3])/x_pool[i][0][3]
print(righta)
print(wronga)
print(rightb)
print(wrongb)
print(ra/righta)
print(wa/wronga)
print(rb/rightb)
print(wb/wrongb)
print(total)
wa321 = np.array(wa123)
ga321 = np.array(ga123)
print(np.corrcoef(wa321,ga321))