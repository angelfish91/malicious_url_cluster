import sys
import os
import json
import pandas as pd



def get_malicious(file_list):
    root_path = "/home/machangchun/sparrow/data/sangfor/18"
    df_list_ma = []
    df_list_ph = []
    for i in file_list:
        local_path = root_path + i
        
        filepath = os.path.join(local_path ,'dumped_result.ma.csv')
        df_list_ma.append( pd.read_csv(filepath))
        
        filepath = os.path.join(local_path ,'dumped_result.ph.csv')
        df_list_ph.append( pd.read_csv(filepath))
        
        print local_path
        
    df_ma = pd.concat(df_list_ma, axis = 0)
    df_ph = pd.concat(df_list_ph, axis = 0)
    return df_ma, df_ph

def get_total_sample(file_list):
    root_path = "/home/machangchun/sparrow/data/sangfor/18"
    total_url = set()
    for i in file_list:
        local_path = root_path + i
        filepath = os.path.join(local_path ,'dumped_result.txt')
        f = open(filepath)
        recs = [json.loads(rec) for rec in f]
        for rec in recs:
            for k, v in rec.iteritems():
                rec[k] = json.loads(v)
        for rec in recs:
            total_url.add(rec['url'])
        print filepath, len(total_url)
        f.close()
    print "Total URL Count:\t%d" %len(total_url)
    return total_url
            
        
        
if __name__ == "__main__":
    file_list = ["0301","0302","0306","0307","0311","0312","0313","0314","0315","0318","0319",
              "0320","0321","0322","0323","0325","0326","0327","0328","0330","0401","0402"]
    
    df_ma, df_ph = get_malicious(file_list)
    total_url = get_total_sample(file_list)
    safe_url = list(set(total_url) - set(df_ma.url) - set(df_ph.url))
    safe_url = [_.encode('utf-8') for _ in safe_url]
    df_safe = pd.DataFrame({'url': safe_url})
    print "Safe URL Count:\t%d" %len(df_safe)
    df_safe.to_csv("/home/machangchun/sparrow/data/sangfor/safe.csv")
    
    
    
    
    
    
    
    
    
    