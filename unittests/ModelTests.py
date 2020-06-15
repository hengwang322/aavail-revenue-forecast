#!/usr/bin/env python
"""
model tests
"""


import unittest

## import model specific functions and variables
from model import *

class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """

    def test_01_train(self):
        """
        test the train functionality
        """
        saved_model = 'models/test-united_kingdom-0_1.joblib'
        ## train the model
        model_train(data_dir='data/cs-train',test=True)
        self.assertTrue(os.path.exists(saved_model))

    def test_02_load(self):
        """
        test the train functionality
        """

        ## train the model
        all_data, all_models = model_load(training=False)

        country_list = ['all', 'eire', 'france', 'germany', 'hong_kong', 'netherlands',
                        'norway', 'portugal', 'singapore', 'spain', 'united_kingdom']
        self.assertEqual(list(all_data.keys()),country_list)
        self.assertEqual(list(all_models.keys()),country_list)


    def test_03_predict(self):
        """
        test the predict function input
        """

        country='all'
        year='2018'
        month='01'
        day='05'

        result = model_predict(country,year,month,day,all_models=None,test=True,from_pickle=False)

        y_pred = result['y_pred']
        self.assertEqual(len(y_pred),1)


### Run the tests
if __name__ == '__main__':
    unittest.main()
