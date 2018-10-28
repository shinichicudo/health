from pymongo import MongoClient
import datetime
import time
import threading
import traceback
import logconfig as lc
import data_deal as dd


# 第三方模块
from PyQt5.QtCore import QTimer
import logging
class MongoEngine(object):
    conn = MongoClient('localhost', 27017)
    db = conn.mydb  #连接mydb数据库，没有则自动创建
    collection = db.daily

    def findLast(self,dict,last_count):
        mongo_data = self.collection.find(dict).sort([("time",-1)]).limit(last_count)
        return self.com_data_deal(mongo_data)

    def find_selective(self,dict):
        mongo_data = self.collection.find(dict)
        return self.com_data_deal(mongo_data)
    def com_data_deal(self,mongo_data):
        a = []
        for row in mongo_data:
            a.append(row)
        return a

    def getAllData(self):
        mongo_data = self.collection.find()
        return self.com_data_deal(mongo_data)

    def update_chance_type(self,dict,chanceType):
        self.collection.update(dict,{'$set' : {'chanceType':chanceType}})

    def find_one(self,dict):
        return self.collection.find_one(dict)


    def find_post_one(self,instrumentID,time):
        list = self.collection.find({'instrumentID':instrumentID,'time':{'$gt':time}}).sort([('time',1)])
        try:
            return list.next()
        except StopIteration:
            return None
    def find_pos_pre_list(self,instrumentID,time,last_count):
        list = self.collection.find({'instrumentID':instrumentID,'time':{'$lte':time}}).sort([('time',-1)]).limit(last_count)
        return self.com_data_deal(list)

if __name__ == '__main__':
    a = {'commodity':'ag'}
    mongo_engine = MongoEngine()
    list = mongo_engine.find_post_one('m0807','2003-04-07')
    print(list)
    # x_list,y_list,t_list,i_list = dd.list_data2train_data(list)
    # print(x_list)




