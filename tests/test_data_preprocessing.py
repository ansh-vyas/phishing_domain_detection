import unittest
from src.data_preprocessing import load_data, preprocess_data

class TestDataPreprocessing(unittest.TestCase):
    def test_load_data(self):
        data = load_data('data/dataset_full.csv')
        self.assertIsNotNone(data)

    def test_preprocess_data(self):
        data = load_data('data/dataset_full.csv')
        X_train, X_test, y_train, y_test = preprocess_data(data)
        self.assertEqual(X_train.shape[0], y_train.shape[0])

if __name__ == '__main__':
    unittest.main()
