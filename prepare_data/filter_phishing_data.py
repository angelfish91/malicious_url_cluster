import json
import csv
import cStringIO
import codecs
import os

from random import shuffle
from urlparse import urlparse

UNAVALIABLE = 'UNAVALIABLE'


def get_pa(rec):
    try:
        pa_result = rec['S009']
    except KeyError:
        return UNAVALIABLE
    return pa_result['Category'] if pa_result else None


def get_bd(rec):
    try:
        bd_result = rec['S012']
    except KeyError:
        return (UNAVALIABLE, UNAVALIABLE)
    return (bd_result['main'], bd_result['sub']) if bd_result else (None, None)


def get_sophos(rec):
    try:
        sophos_result = rec['S010']
    except KeyError:
        return UNAVALIABLE
    return sophos_result['categoryname'] if sophos_result else None


def get_gsb(rec):
    try:
        gsb_result = rec['S000']
    except KeyError:
        return UNAVALIABLE
    if gsb_result:
        rv = gsb_result['threatType']
        return tuple(sorted(rv))
    else:
        return None


SOPHOS_AC = set([
    'Browser Exploits', 'Phishing', 'Malicious Sites',
    'Hacking/Computer Crime'])


def sample(grp, sp_ratio):
    sp_grp = {}
    for p, urls in grp.iteritems():
        sp_sz = int(sp_ratio * len(urls))
        if sp_sz == 0:
            sp_sz = 1
        shuffle(urls)
        sp_grp[p] = urls[0:sp_sz]
    return sp_grp


def write_csv(grp, file_path):
    with open(file_path, 'wb') as f:
        f_csv = UnicodeWriter(f)
        f_csv.writerow([
            'url', 'freq', 'dev_cnt', 'platform', 'report_type', 'module_type',
            'sophos', 'pa', 'baidu_main', 'baidu_sub', 'gsb'])
        for p, urls in grp.iteritems():
            for url in urls:
                f_csv.writerow(list(url) + list(p))


def has_phishing(p):
    sophos_cat, pa_cat, bd_main_cat, bd_sub_cat, gsb_cat = p
    return (
            (sophos_cat and (sophos_cat == 'Phishing')) or
            (pa_cat and (pa_cat == 'phishing')) or
            (bd_main_cat and (bd_main_cat == 3)) or
            (gsb_cat and ('SOCIAL_ENGINEERING' in gsb_cat)))


def dedup_by_hostname(recs):
    hostnames = set()
    deduped_recs = []
    for rec in recs:
        hostname = urlparse(rec[0]).hostname
        if hostname not in hostnames:
            hostnames.add(hostname)
            deduped_recs.append(rec)
    return deduped_recs


def has_malware(p):
    sophos_cat, pa_cat, bd_main_cat, bd_sub_cat, gsb_cat = p
    return (
            (sophos_cat and (sophos_cat in [
                'Hacking/Computer Crime', 'Malicious Downloads',
                'Spyware/adware'])) or
            (pa_cat and (pa_cat == 'malware')) or
            (bd_main_cat and (bd_main_cat == 4)) or
            (gsb_cat and ('MALWARE' in gsb_cat)))


class UnicodeWriter:
    """
    A CSV writer which will write rows to CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        # Redirect output to a queue
        self.queue = cStringIO.StringIO()
        self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        for i in xrange(0, len(row)):
            if \
                    (not row[i]) or \
                    (isinstance(row[i], (int, tuple))):
                row[i] = str(row[i])
        self.writer.writerow([s.encode("utf-8") for s in row])
        # Fetch UTF-8 output from the queue ...
        data = self.queue.getvalue()
        data = data.decode("utf-8")
        # ... and reencode it into the target encoding
        data = self.encoder.encode(data)
        # write to the target stream
        self.stream.write(data)
        # empty queue
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


def add_grp(grp, url, p):
    urls = grp.get(p)
    if urls:
        urls.append(url)
    else:
        grp[p] = [url]


def get_cats(rec):
    bd_main, bd_sub = get_bd(rec)
    return (get_sophos(rec), get_pa(rec), bd_main, bd_sub, get_gsb(rec))


def worker(path):
    f = open(path)
    recs = (json.loads(rec) for rec in f)
    grp = {}
    for rec in recs:
        for k, v in rec.iteritems():
            rec[k] = json.loads(v)
        # if 'S012' not in rec or 'S010' not in rec or 'S000' not in rec:
        #     continue
        if rec['private_data'].get('task_name') == 'ZeroVirusURLMining':
            continue
        k = get_cats(rec)
        stat = rec['private_data']['stat'].split(',')
        if has_phishing(k):
            add_grp(grp, [rec['url']] + list(stat), k)
    f.close()
    return grp

if __name__ == "__main__":
    
    root_path = "/home/sparrow/sparrow/data/sangfor/18"
    
    for i in ["0417", "0418", "0419"]:
        local_path = root_path + i
        file_path = os.path.join(local_path, "dumped_result.txt")
        output_filepath = os.path.join(local_path ,'dumped_result.ph.csv')
        print output_filepath
        if os.path.isfile(output_filepath):
            continue
        grp = worker(file_path)
        write_csv(grp, output_filepath)
        