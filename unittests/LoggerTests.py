#!/usr/bin/env python
"""
model tests
"""

import os
import csv
import unittest
from ast import literal_eval
import pandas as pd

## import model specific functions and variables
from logger import update_train_log, update_predict_log

class LoggerTest(unittest.TestCase):
    """
    test the essential functionality
    """

    def test_01_train(self):
        """
        ensure log file is created
        """

        log_file = os.path.join("logs","all-train-test.log")
        if os.path.exists(log_file):
            os.remove(log_file)

        ## update the log
        tag = 'all'
        dt_range = ('2018-01-01','2018-02-01')
        eval_test = {'rmse':0.5}
        runtime = "00:00:01"
        MODEL_VERSION = 0.1
        MODEL_VERSION_NOTE = "test model"

        update_train_log(tag,dt_range,eval_test,runtime,MODEL_VERSION,MODEL_VERSION_NOTE,test=True)

        self.assertTrue(os.path.exists(log_file))

    def test_02_train(self):
        """
        ensure that content can be retrieved from log file
        """

        log_file = os.path.join("logs","all-train-test.log")

        ## update the log
        tag = 'all'
        dt_range = ('2018-01-01','2018-02-01')
        eval_test = {'rmse':0.5}
        runtime = "00:00:01"
        MODEL_VERSION = 0.1
        MODEL_VERSION_NOTE = "test model"

        update_train_log(tag,dt_range,eval_test,runtime,MODEL_VERSION,MODEL_VERSION_NOTE,test=True)

        df = pd.read_csv(log_file)
        logged_eval_test = [literal_eval(i) for i in df['eval_test'].copy()][-1]
        self.assertEqual(eval_test,logged_eval_test)


    def test_03_predict(self):
        """
        ensure log file is created
        """

        log_file = os.path.join("logs","predict-test.log")
        if os.path.exists(log_file):
            os.remove(log_file)

        ## update the log
        country = 'ERIE'
        y_pred = [15000]
        y_proba = None
        target_date = '2018-01-05'
        runtime = "00:00:02"
        MODEL_VERSION = 0.1

        update_predict_log(country,y_pred,y_proba,target_date,runtime, MODEL_VERSION, test=True)

        self.assertTrue(os.path.exists(log_file))


    def test_04_predict(self):
        """
        ensure that content can be retrieved from log file
        """

        log_file = os.path.join("logs","predict-test.log")

        ## update the log
        country = 'ERIE'
        y_pred = [15000]
        y_proba = None
        target_date = '2018-01-05'
        runtime = "00:00:02"
        MODEL_VERSION = 0.1

        update_predict_log(country,y_pred,y_proba,target_date,runtime, MODEL_VERSION, test=True)

        df = pd.read_csv(log_file)
        logged_y_pred = [literal_eval(i) for i in df['y_pred'].copy()][-1]
        self.assertEqual(y_pred,logged_y_pred)


### Run the tests
if __name__ == '__main__':
    unittest.main()
