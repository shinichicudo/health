import pandas as pd
import config as cf
def list_data2train_data(list):

    pddata = pd.DataFrame(list)
    pddata_group = pddata.groupby(['instrumentID']).head(1)
    print(pddata_group)

    x_list =[]
    y_list =[]
    t_list =[]
    i_list =[]
    for instrumentID in pddata_group['instrumentID'].tolist():
        pddata_group_ins_sort = pddata.loc[pddata['instrumentID'] == instrumentID].sort_values('time',ascending=False)
        pd_x = pddata_group_ins_sort[["opend","high","low","close"]]
        pd_y = pddata_group_ins_sort[["chanceType"]]
        pd_t = pddata_group_ins_sort[["time"]]
        pd_i = pddata_group_ins_sort[["instrumentID"]]
        pd_v = pddata_group_ins_sort[["volume"]]
        # print(pd_t.count())
        # print(pd_t["time"].count())
        if pd_t["time"].count()< cf.ROW_LENGTH:
            continue
        else:
            for i in range(0,pd_t["time"].count()-cf.ROW_LENGTH+1):
                v_p = pd_v[i:i+cf.ROW_LENGTH].sum().tolist()
                if v_p[0]<cf.LEAST_VOLUME_SIZE:
                    continue
                x_p = pd_x[i:i+cf.ROW_LENGTH].values.tolist()
                y_p = pd_y.iat[i,0]
                t_p = pd_t.iat[i,0]
                i_p = pd_i.iat[i,0]
                x_list.append(x_p)
                y_list.append(y_p)
                t_list.append(t_p)
                i_list.append(i_p)
    return (x_list,y_list,t_list,i_list)