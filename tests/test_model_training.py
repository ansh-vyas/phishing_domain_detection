import unittest
import os
from src.model_training import train_model

class TestModelTraining(unittest.TestCase):

    def setUp(self):
        # Path where model is saved
        self.model_path = 'models/phishing_detection_model.pkl'

    def test_train_model(self):
        # Train model
        model, accuracy = train_model()

        # Verify model object and accuracy score are returned
        self.assertIsNotNone(model)
        self.assertTrue(isinstance(accuracy, float))

        # Check if model file is saved
        self.assertTrue(os.path.exists(self.model_path))

    def tearDown(self):
        # Clean up by removing the model file after testing
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

if __name__ == '__main__':
    unittest.main()
