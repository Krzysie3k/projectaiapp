from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# Funkcja do przetwarzania tekstu
def preprocess_text(text):
    text = text.lower()  # Zamiana na małe litery
    text = re.sub(r'\d+', '', text)  # Usunięcie liczb
    text = re.sub(r'[^\w\s]', '', text)  # Usunięcie znaków specjalnych
    return text

# Ładowanie modelu i wektoryzatora
model_filename = 'model/spam_model.pkl'
vectorizer_filename = 'model/vectorizer.pkl'

model = pickle.load(open(model_filename, 'rb'))
vectorizer = pickle.load(open(vectorizer_filename, 'rb'))

# Funkcja do przewidywania, czy tekst jest spamem
def predict_spam(text):
    text = preprocess_text(text)
    text_vect = vectorizer.transform([text])
    prediction = model.predict(text_vect)
    return prediction[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        sms_text = request.form['sms_text']
        prediction = predict_spam(sms_text)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
