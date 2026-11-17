
from model import predict_wine_quality

sample_wine = {
    "fixed acidity": 7.0,
    "volatile acidity": 0.27,
    "citric acid": 0.36,
    "residual sugar": 20.7,
    "chlorides": 0.045,
    "free sulfur dioxide": 45.0,
    "total sulfur dioxide": 170.0,
    "density": 1.0010,
    "pH": 3.00,
    "sulphates": 0.45,
    "alcohol": 8.8,
}


print("\n=== WHITE WINE QUALITY PREDICTION ===")

result = predict_wine_quality(sample_wine)

print(f"Prediction: {result['prediction']}")
print(f"Class ID: {result['class_id']}")
print(f"Confidence: {result['confidence']:.2%}")