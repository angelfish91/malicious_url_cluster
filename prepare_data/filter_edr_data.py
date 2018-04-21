import json
import csv
import cStringIO
import codecs
import os
import pandas as pd

from random import shuffle
from urlparse import urlparse

UNAVALIABLE = 'UNAVALIABLE'

def get_edr(data):
    data = data["S104"]
    attr = data["attr"]
    file_size = data["file_size"]
    file_type = data["file_type"]
    return [attr, file_size, file_type]
    

def raw_data_parser(path):
    def get_list(lst, n):
        return [_[n] for _ in lst]
    f = open(path)
    recs = (json.loads(rec) for rec in f)
    data = []
    for rec in recs:
        
        for k, v in rec.iteritems():
            rec[k] = json.loads(v)
        # if 'S012' not in rec or 'S010' not in rec or 'S000' not in rec:
        #     continue
        if rec['private_data'].get('task_name') == 'ZeroVirusURLMining':
            continue
        url = [rec["url"]]
        try:
            edr = get_edr(rec)
        except:
            continue
        data.append(url+edr)
    f.close()
    df = pd.DataFrame({"url":get_list(data, 0), "attr":get_list(data, 1), 
                       "file_size":get_list(data, 2), "file_type":get_list(data, 3)})
    df = df.loc[df.file_type != 0]
    dump_path = os.path.join(os.path.dirname(path), "dumped_result.edr.csv")
    df.to_csv(dump_path, index = False)
    print dump_path
    return df


def raw_data_parser_wapper(file_list):
    for i in file_list:
        path = "/home/sparrow/sparrow/data/sangfor/18" + i
        path = os.path.join(path, "dumped_result.txt")
        raw_data_parser(path)
    return

if __name__ == "__main__":
    file_list = ["0401", "0402"]
    raw_data_parser_wapper(file_list)
    
    
    
