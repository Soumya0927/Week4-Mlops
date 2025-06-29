import unittest
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

class TestModelAccuracy(unittest.TestCase):
    def test_model_accuracy_on_sample_data(self):
        # Load the model
        model = joblib.load("models/models.pkl")
        
        # Load the sample CSV
        df = pd.read_csv("samples.csv")

        # Prepare features and labels
        X = df.drop(columns=["species"])
        y_true = df["species"]

        # Predict using the model
        y_pred = model.predict(X)

        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")

        # Assert that the accuracy is 100%
        self.assertEqual(accuracy, 1.0, "Model accuracy is not 100% on sample data")

if __name__ == "__main__":
    unittest.main()
