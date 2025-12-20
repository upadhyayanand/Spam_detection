import joblib

model = joblib.load("best_model.pkl")

def predict_message(message):
    return model.predict([message])[0]


# Test
print(predict_message("Free cashback offer just for you"))
print(predict_message("Meeting scheduled tomorrow at 10 am"))
