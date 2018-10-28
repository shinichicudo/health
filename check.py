from mongo_engine import MongoEngine
import numpy as np
from histogram import Histogram
import config as cf

if __name__ == '__main__':
    mongo = MongoEngine()
    a = mongo.find_selective({'chanceType':1})
    b = mongo.find_selective({'chanceType':-1})
    c = mongo.find_selective({'chanceType':2})
    d = np.concatenate((a,b,c))
    print(d)

    for dd in d:
        list = mongo.find_pos_pre_list(dd['instrumentID'],dd['time'],cf.ROW_LENGTH)
        x = []
        for l in list:
            xx =[]
            xx.append(l['opend'])
            xx.append(l['high'])
            xx.append(l['low'])
            xx.append(l['close'])
            x.append(xx)
        x = np.array(x)
        y = list[0]['chanceType']
        time = list[0]['time']
        instrumentID = list[0]['instrumentID']
        ha = (x,y,time,instrumentID)
        histo = Histogram(ha)
        histo.createImage()
    # e = np.delete(d,[])