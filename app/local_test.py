import numpy as np
from joblib import load

test_np_array = np.array([[1], [2], [17]])

try:
    model = load('model.joblib')
    preds = model.predict(test_np_array)
    preds_as_str = str(preds)
    print(preds_as_str)

except Exception as e:
    print(f"An error occurred: {e}")