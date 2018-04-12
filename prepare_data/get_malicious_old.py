import pandas as pd
import numpy as np
import json
import sys
from collections import Counter




def load_malicious_data(path):
    with open(path, 'r') as f:
        res = []
        for line in f:
            res.append(json.loads(line.strip()))
            
    res = [ _[list(_.keys())[0]] for _ in res]
    class_type = [_['class_type'] for _ in res]
    is_domain = [_['is_domain'] for _ in res]
    urls = [_['url'] for _ in res]
    
    df = pd.DataFrame({'url':urls, 'is_domain':is_domain,"class_type":class_type})
    return df

if __name__ == "__main__":
    df_list = []
    for i in range(1,5):
        path = "../../data/sangfor_old/report_malice_link_succe_{i}.txt".format(i=str(i))
        df = load_malicious_data(path)
        df_list.append(df)
        print path
    df = pd.concat(df_list, axis = 0)
    df = df.drop_duplicates("url")
    print "url count", len(df)
    df.to_csv("../../data/sangfor_old/malicious.csv", index = False)