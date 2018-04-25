#!/usr/bin/evn python2.7
# -*- coding: utf-8 -*-
"""
using google to vlidate the results
"""
import os
import time
import requests
import urllib
import hashlib
import pandas as pd
from logger import logger
from joblib import Parallel, delayed
from collections import Counter

TIMEOUT = 60

def timelimit(timeout, func, args=(), kwargs={}):
    """ Run func with the given timeout. If func didn't finish running
        within the timeout, raise TimeLimitExpired
    """
    import threading
    class FuncThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            self.result = func(*args, **kwargs)

        def _stop(self):
            if self.isAlive():
                threading.Thread._Thread__stop(self)

    it = FuncThread()
    it.start()
    it.join(timeout)
    if it.isAlive():
        it._stop()
        raise IOError
    else:
        return it.result


def core_download(index, url, file_path):
    try:
        timelimit(TIMEOUT, urllib.urlretrieve,(url, file_path))
        with open(file_path, 'rb') as f:
            data = f.read()
        sha256 = hashlib.sha256(data).hexdigest()
        logger.debug("index\t%d\t%s\t%s" % (index, url, sha256))
        return sha256
    except Exception as e:
        print ("Downloading File Fail %s\t%s" % (url, str(e)))
        return None


class ValidateWithVT(object):
    def __init__(
            self,
            n_jobs=8,
            api_key='86328d300ba0498ef2b0ad322ab8c4dd5aa62a1c16ece6946b3f541f0b83ecd1',
            time_out=16):
        self.api_key = api_key
        self.time_out = time_out
        self.n_jobs = n_jobs
        self.headers = {
            "Accept-Encoding": "gzip, deflate",
            "User-Agent": "gzip,  My Python requests library example client or username"}

    def download_file(self, urls, output_path):
        assert isinstance(urls, list) or isinstance(urls, str)
        if isinstance(urls, str):
            urls = [urls]
        if not os.path.isdir(output_path):
            try:
                os.mkdir(output_path)
                logger.debug("mkdir %s" % output_path)
            except Exception as e:
                logger.error("mkdir error %s %s" % (output_path, str(e)))
                return
        file_path_list = [
            os.path.join(
                output_path,
                "{i}.exe".format(
                    i=_)) for _ in range(
                len(urls))]
        sha256_list = Parallel(
            n_jobs=self.n_jobs)(
            delayed(core_download)(
                index,
                url,
                file_path) for index, url, file_path in zip(
                range(len(urls)),
                urls,
                file_path_list))
        df = pd.DataFrame(
            {"url": urls, "path": file_path_list, "sha256": sha256_list})
        return df

    
    @staticmethod
    def calc_sha256(file_dir):
        assert isinstance(file_dir, str)
        if os.path.isdir(file_dir):
            file_name = os.listdir(file_dir)
            file_dir = [os.path.join(file_dir, _) for _ in file_name]
        elif os.path.isfile(file_dir):
            file_dir = [file_dir]
        else:
            raise ValueError("file dir is illegal %s" % file_dir)
        sha256_list = []
        for file_path in file_dir:
            with open(file_path, "rb") as f:
                data = f.read()
            sha256_list.append(hashlib.sha256(data).hexdigest())
        df = pd.DataFrame({"file_path": file_dir, "sha256": sha256_list})
        return df, Counter(sha256_list)

    def validate_url_with_vt(self, urls):
        test_res = []
        for index, test_url in enumerate(urls):
            params = {'apikey': self.api_key, 'resource': test_url}
            try:
                response = requests.post(
                    'https://www.virustotal.com/vtapi/v2/url/report',
                    params=params,
                    headers=self.headers)
                json_response = response.json()
                test_res.append(json_response['positives'])
            except Exception as e:
                logger.error("%s" % str(e))
                test_res.append(-1)
            logger.debug("Index %d\tPositives %d\t %s" %
                         (index, test_res[-1], test_url))
            time.sleep(self.time_out)
        df = pd.DataFrame({"url": urls, "positives": test_res})
        return df

    def upload_file_to_vt(self, file_path):
        assert isinstance(file_path, str)
        if os.path.isdir(file_path):
            file_dir = file_path
            file_list = os.listdir(file_dir)
            file_path = [os.path.join(file_dir, _) for _ in file_list]
        elif os.path.isfile(file_path):
            file_path = [file_path]
        else:
            raise ValueError("file value error %s" % file_path)

        params = {'apikey': self.api_key}
        sha256_list = []
        for index, each_file in enumerate(file_path):
            df, counter = self.calc_sha256(each_file)
            sha256 = list(df.sha256)[0]
            if sha256 in sha256_list:
                continue
            files = {
                'file': (
                    '{index}.exe'.format(
                        index=index), open(
                        each_file, 'rb'))}
            try:
                response = requests.post(
                    'https://www.virustotal.com/vtapi/v2/file/scan',
                    files=files,
                    params=params)
                json_response = response.json()
                sha256_list.append(json_response['sha256'])
                logger.debug("success to upload file to VT %s" %sha256_list[-1])
            except Exception as e:
                logger.error("Upload file to VT error %s" % e)
        return sha256_list

    def validate_file_with_vt(self, sha256, verbose=False):
        assert isinstance(sha256, list) or isinstance(sha256, str)
        if isinstance(sha256, list):
            sha256_list = sha256
        else:
            sha256_list = [sha256]
        positives = []
        for index, sha256 in enumerate(sha256_list):
            params = {'apikey': self.api_key, 'resource': sha256}
            try:
                response = requests.get(
                    'https://www.virustotal.com/vtapi/v2/file/report',
                    params=params,
                    headers=self.headers)
                json_response = response.json()
                positives.append(json_response['positives'])
                if verbose:
                    logger.debug("%s" % str(json_response))
            except Exception as e:
                logger.error("get VT file report fail %s" % str(e))
                positives.append(-1)
            logger.debug("index %d\t sha256 %s\tpositives %d" %
                         (index, sha256, positives[-1]))
            time.sleep(self.time_out)
        df = pd.DataFrame({"sha256": sha256_list, "positives": positives})
        return df
