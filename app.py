from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# load model & scaler
model = pickle.load(open('svm_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    label = None

    if request.method == 'POST':
        features = np.array([[  
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]])

        features_scaled = scaler.transform(features)

        # ===== PREDIKSI =====
        label = int(model.predict(features_scaled)[0])

        # ===== PROBABILITAS =====
        proba = model.predict_proba(features_scaled)[0]
        probability = round(proba[label] * 100, 2)

        if label == 1:
            prediction = "BERISIKO Penyakit Jantung"
        else:
            prediction = "TIDAK Berisiko Penyakit Jantung"

    return render_template(
        'index.html',
        prediction=prediction,
        probability=probability,
        label=label
    )

if __name__ == '__main__':
    app.run(debug=True)
