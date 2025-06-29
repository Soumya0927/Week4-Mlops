import unittest
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

class TestModelAccuracy(unittest.TestCase):
    def test_model_accuracy_on_sample_data(self):
        # Load model and label encoder
        model = joblib.load("models/models.pkl")

        # Load the sample data
        df = pd.read_csv("samples.csv")

        # Features and true labels
        X = df.drop(columns=["species"])
        y_true = df["species"]

        # Predict using model (returns encoded labels)
        y_pred_encoded = model.predict(X)

        # Map encoded predictions to actual labels
        label_encoder = LabelEncoder()
        label_encoder.fit(["setosa", "versicolor", "virginica"])
        y_pred = label_encoder.inverse_transform(y_pred_encoded)

        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")

        # Assert that the accuracy is 100%
        self.assertEqual(accuracy, 1.0, "Model accuracy is not 100% on sample data")

if __name__ == "__main__":
    unittest.main()

