

import scrapy
from pymongo import MongoClient
import datetime
import json
class TestSpider(scrapy.spiders.Spider):
    name = "testspider"
    allowed_domains = ["testspider.org"]
    future_list =['hc','bu','zn','ru','al','cu','rb','ni','sn','p','pp','jd','i','jm','v','l','y','c','m','j','cs','ZC','FG','MA','CF','RM','TA','SR','ag','au','b','AP']
    # future_list=['ag']
    start_urls = []
    base_url_daily = "http://stock2.finance.sina.com.cn/futures/api/json.php/IndexService.getInnerFuturesDailyKLine?symbol="
    # base_url_daily = "http://stock2.finance.sina.com.cn/futures/api/json.php/IndexService.getInnerFuturesMiniKLine15m?symbol="

    # mongoDB
    conn = MongoClient('localhost', 27017)
    db = conn.mydb  #连接mydb数据库，没有则自动创建
    end_date= ''
    collection = db.daily
    date_save = db.datesave
    start_date = ''
    start_year = '18'
    start_month ='01'
    end_month = '13'


    def __init__(self):
        sd = self.date_save.find_one({'the':1})
        if sd is not None:
            self.start_date = sd['time']
            self.start_year = self.start_date.split('-')[0][2:4]
        else:
            self.start_year = '08'

        self.start_month ='01'
        self.end_month = '13'

        self.end_date = (datetime.datetime.now() + datetime.timedelta(weeks=104)).strftime('%Y-%m-%d')
        end_year = self.end_date.split('-')[0][2:4]

        new_start_date = datetime.datetime.now().strftime('%Y-%m-%d')
        dict = {'the':1}
        sd = self.date_save.find_one(dict)
        if sd is not None:
            self.date_save.update(dict,{'$set' : {'time':new_start_date}})
        else:
            dict['time']=new_start_date
            self.date_save.insert(dict)


        for  i in range(len(self.future_list)):
            current_year = self.start_year
            while(int(current_year) != int(end_year)):
                current_month = self.start_month
                while(int(current_month)!=int(self.end_month)):
                    symbol =self.future_list[i] + current_year + current_month
                    self.start_urls.append(self.base_url_daily+symbol)
                    current_month = str(int(current_month) + 1).zfill(2)
                current_year = str(int(current_year) + 1).zfill(2)

    def parse(self, response):
        instrumentID = response.url.split("symbol=")[1]
        commodity = instrumentID[:-4]
        data = str(response.body, encoding = "utf-8")
        if data == 'null':
            return
        else:
            ar = json.loads(data)
            array = []
            for a in ar:
                if a[0]>=self.start_date:
                    array.append(a)
            for i in range(len(array)):
                dic = {}
                dic['instrumentID'] = instrumentID
                dic['commodity'] = commodity
                dic['time'] = array[i][0]
                if dic['time'] == self.start_date:
                    self.collection.delete_one(dic)

                for j in range(len(array[i])):
                    if j == 0:
                        None
                    elif j == 1:
                        dic['opend'] = float(array[i][j])
                    elif j == 2:
                        dic['high'] = float(array[i][j])
                    elif j == 3:
                        dic['low'] = float(array[i][j])
                    elif j == 4:
                        dic['close'] = float(array[i][j])
                    elif j == 5:
                        dic['volume'] = int(array[i][j])
                dic['chanceType'] = 0
                self.collection.insert(dic)
if __name__ == '__main__':
    # mongoDB
    conn = MongoClient('localhost', 27017)
    db = conn.mydb  #连接mydb数据库，没有则自动创建
    daily = db.daily
    sd = db.datesave

    # s = sd.find_one({"time" : "2018-08-15"})
    # sd.update({'_id':s['_id']},{'$set' : {'time':'2018-08-10'}})
    # a = {'chanceType':8}
    # c = daily.update(a,{'$set' : {'chanceType':2}})
    a = {'instrumentID':'ag1812'}
    c = daily.find(a)

    print(c['time']+ "  "+c['instrumentID']+ "  "+c['commodity']+ "  "+str(c['chanceType'])+ "  "+str(c['opend'])+ "  "+str(c['high'])+ "  "+str(c['low'])+ "  "+str(c['close'])+ "  ")
    daily.update({'_id':c['_id']},{'$set' : {'chanceType':0}})
    c = daily.find_one(a)
    print(c['time']+ "  "+c['instrumentID']+ "  "+c['commodity']+ "  "+str(c['chanceType'])+ "  "+str(c['opend'])+ "  "+str(c['high'])+ "  "+str(c['low'])+ "  "+str(c['close'])+ "  ")

    # #

    # for c in fifteen.find():
    #     fifteen.update({'_id':c['_id']},{'$set' : {'orderType':0}})
    # future_list =['hc','bu','zn','ru','al','cu','rb','ni','sn','p','pp','jd','i','jm','v','l','y','c','m','j','cs','ZC','FG','MA','CF','RM','TA','SR']
    # start_urls = []
    # start_year = '08'
    # start_month ='01'
    # end_month = '13'
    #
    # end_date = (datetime.datetime.now() + datetime.timedelta(weeks=104)).strftime('%Y-%m-%d')
    # end_year = end_date.split('-')[0][2:4]
    #
    #
    #
    # for  i in range(len(future_list)):
    #     current_year = start_year
    #     while(int(current_year) != int(end_year)):
    #         current_month = start_month
    #         while(int(current_month)!=int(end_month)):
    #             symbol =future_list[i] + current_year + current_month
    #             start_urls.append(symbol)
    #             current_month = str(int(current_month) + 1).zfill(2)
    #         current_year = str(int(current_year) + 1).zfill(2)
    #
    # a = json.loads(json.dumps(start_urls))
    # print(start_urls)