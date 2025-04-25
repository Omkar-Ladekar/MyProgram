
import joblib

model = joblib.load("model/model.pkl")
text = input("Enter a customer message: ")
prediction = model.predict([text])[0]
print(f"\n🔍 Predicted category: {prediction}")
