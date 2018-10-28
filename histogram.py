import matplotlib.pyplot as plt
import mpl_finance as mpf
from matplotlib.pylab import date2num
import datetime
from matplotlib.widgets import Button
from mongo_engine import MongoEngine
import config as cf
import data_deal as dd
class Histogram(object):



    up_buttion = [0.91,0.8,0.08,0.03]
    down_buttion = [0.91,0.7,0.08,0.03]
    damped_buttion = [0.91,0.6,0.08,0.03]
    not_order_buttion = [0.91,0.5,0.08,0.03]
    mongo_engine = MongoEngine()
    x = []
    y = 0
    time = 0
    instrumentID = ''
    button_dict = {}



    def __init__(self,d):
        self.result = cf.NONE_SENSE_DATA_TYPE
        self.data_list = []
        self.x,self.y,self.time,self.instrumentID = d
        self.button_dict = {'time':self.time,'instrumentID':self.instrumentID}
        flag = 0
        for data in self.x:
            if flag ==0:
                mongo_engine = MongoEngine()
                right_data = mongo_engine.find_selective({'instrumentID':self.instrumentID,'time':self.time})
                if right_data[0]['opend'] !=data[0] and right_data[0]['high'] !=data[1] and right_data[0]['low'] !=data[2] and right_data[0]['close'] !=data[3]:
                    print(right_data[0])

                    wrong_data = mongo_engine.find_selective({'opend':data[0],'high':data[1],'low':data[2],'close':data[3]})
                    print(wrong_data[0])
                    raise NameError
                date_time = datetime.datetime.strptime(self.time,'%Y-%m-%d')
                t = date2num(date_time)
            flag = 1
            open,high,low,close = (data[0],data[1],data[2],data[3])
            datas = (t,open,high,low,close)
            self.data_list.append(datas)
            t -= 1

    def draw_button_up(self,axis):
        global button_up#must global
        point = plt.axes(axis)
        button_up = Button(point, 'up')
        button_up.on_clicked(self.button_press_up)

    def draw_button_down(self,axis):
        global button_down#must global
        point = plt.axes(axis)
        button_down = Button(point,'down')
        button_down.on_clicked(self.button_press_down)

    def draw_button_damped(self,axis):
        global button_damped#must global
        point = plt.axes(axis)
        button_damped = Button(point,'damped')
        button_damped.on_clicked(self.button_press_damped)

    def draw_button_not_order(self,axis):
        global button_not_order#must global
        point = plt.axes(axis)
        button_not_order = Button(point,'not_order')
        button_not_order.on_clicked(self.button_press_not_order)

    def button_press_up(self,event):
        print('button up is pressed!')
        d = {'instrumentID':self.instrumentID,'time':self.time}
        o = self.mongo_engine.find_one(d)
        m = self.mongo_engine.find_post_one(self.instrumentID,self.time)
        if m is not None and o['close']-m['close']<0:
            self.mongo_engine.update_chance_type(self.button_dict,1)
            self.result = 1
        else:
            self.mongo_engine.update_chance_type(self.button_dict,0)
            self.result = 0
    def button_press_down(self,event):
        print('button down is pressed!')
        d = {'instrumentID':self.instrumentID,'time':self.time}
        o = self.mongo_engine.find_one(d)
        m = self.mongo_engine.find_post_one(self.instrumentID,self.time)
        print(m)
        if m is not None and o['close']-m['close']>0:
            self.mongo_engine.update_chance_type(self.button_dict,-1)
            self.result = -1
        else:
            self.mongo_engine.update_chance_type(self.button_dict,0)
            self.result = 0
    def button_press_damped(self,event):
        print('button damped is pressed!')
        d = {'instrumentID':self.instrumentID,'time':self.time}
        o = self.mongo_engine.find_one(d)
        m = self.mongo_engine.find_post_one(self.instrumentID,self.time)
        if m is not None and o['close']-m['close']<0:
            self.mongo_engine.update_chance_type(self.button_dict,2)
            self.result = 2
        else:
            self.mongo_engine.update_chance_type(self.button_dict,0)
            self.result = 0

    def button_press_not_order(self,event):
        print('button not_order is pressed!')
        self.mongo_engine.update_chance_type(self.button_dict,cf.NOT_ORDER_DATA_TYPE)
        self.result = cf.NOT_ORDER_DATA_TYPE


    def createImage(self):
        # 创建子图
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.2)
        fig.set_figheight(7)
        fig.set_figwidth(14)
        # 设置X轴刻度为日期时间
        ax.xaxis_date()
        plt.xticks(rotation=1500)
        plt.yticks()
        plt.title(self.instrumentID)
        plt.xlabel("time")
        plt.ylabel("price")

        self.draw_button_up(self.up_buttion)
        self.draw_button_down(self.down_buttion)
        self.draw_button_damped(self.damped_buttion)
        self.draw_button_not_order(self.not_order_buttion)

        # fig.set_facecolor('green')
        # mpf.index_bar(ax,data_list)
        mpf.candlestick_ohlc(ax,self.data_list,width=0.8,colorup='r',colordown='b')

        plt.grid()

        plt.show()
if __name__ == '__main__':
    a = {'instrumentID':'ag1812'}
    mongo_engine = MongoEngine()
    list = mongo_engine.findLast(a,cf.ROW_LENGTH)
    x_list,y_list,t_list,i_list = dd.list_data2train_data(list)
    data = (x_list[0],y_list[0],t_list[0],i_list[0])
    histogram = Histogram(data)
    histogram.createImage()