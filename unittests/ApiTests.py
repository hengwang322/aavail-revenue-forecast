#!/usr/bin/env python
"""
api tests

these tests use the requests package however similar requests can be made with curl

e.g.

data = '{"key":"value"}'
curl -X POST -H "Content-Type: application/json" -d "%s" http://localhost:8080/predict'%(data)
"""

import sys
import os
import unittest
import requests
import re
from ast import literal_eval
import numpy as np

port = 8080

try:
    requests.post(f'http://127.0.0.1:{port}/predict')
    server_available = True
except:
    server_available = False

## test class for the main window function
class ApiTest(unittest.TestCase):
    """
    test the essential functionality
    """

    @unittest.skipUnless(server_available,"local server is not running")
    def test_01_train(self):
        """
        test the train functionality
        """
        data_dir = 'data/cs-train'
        request_json = {'query':data_dir,'type':'dict','mode':'test'}

        r = requests.post(f'http://127.0.0.1:{port}/train',json=request_json)
        train_complete = re.sub("\W+","",r.text)
        self.assertEqual(train_complete,'true')

    @unittest.skipUnless(server_available,"local server is not running")
    def test_02_predict_empty(self):
        """
        ensure appropriate failure types
        """

        ## provide no data at all
        r = requests.post(f'http://127.0.0.1:{port}/predict')
        self.assertEqual(re.sub('\n|"','',r.text),"[]")

        ## provide improperly formatted data
        r = requests.post(f'http://127.0.0.1:{port}/predict',json={"foo":"bar"})
        self.assertEqual(re.sub('\n|"','',r.text),"[]")

    @unittest.skipUnless(server_available,"local server is not running")
    def test_03_predict(self):
        """
        test the predict functionality
        """
        country='all'
        year='2018'
        month='01'
        day='05'

        query_data = (country,year,month,day)
        query_type = 'dict'
        request_json = {'query':query_data,'type':query_type,'mode':'test'}

        r = requests.post(f'http://127.0.0.1:{port}/predict',json=request_json)
        response = literal_eval(r.text)

        self.assertEqual(1,len(response))

    @unittest.skipUnless(server_available,"local server is not running")
    def test_04_logs(self):
        """
        test the log functionality
        """

        file_name = 'united_kingdom-train-test.log'
        # request_json = {'file':'train-test.log'}
        r = requests.get(f'http://127.0.0.1:{port}/logs/{file_name}')

        with open(file_name, 'wb') as f:
            f.write(r.content)

        self.assertTrue(os.path.exists(file_name))

        if os.path.exists(file_name):
            os.remove(file_name)


### Run the tests
if __name__ == '__main__':
    unittest.main()
