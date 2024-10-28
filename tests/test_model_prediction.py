import unittest
import pandas as pd
from src.model_prediction import predict

class TestModelPrediction(unittest.TestCase):

    def setUp(self):
        # Sample test data
        self.sample_data = pd.DataFrame({
            'qty_dot_url': [2],
            'qty_hyphen_url': [0],
            'length_url': [54]
        })

    def test_predict(self):
        # Make a prediction
        predictions = predict(self.sample_data)

        # Check if predictions are returned in expected format
        self.assertIsInstance(predictions, list)
        self.assertTrue(all(pred in [0, 1] for pred in predictions))

if __name__ == '__main__':
    unittest.main()
