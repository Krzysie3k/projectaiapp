import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# Wczytaj dane
df = pd.read_csv(os.path.join(os.getcwd(), 'spam.csv'), encoding='latin-1')

# Zmiana nazw kolumn
df = df[['label', 'message']].rename(columns={'label': 'label', 'message': 'text'})

# Zbalansowanie danych (jeśli klasy są niezrównoważone)
# Sprawdzamy ile jest próbek w każdej klasie
spam = df[df['label'] == 'spam']
ham = df[df['label'] == 'ham']

# Jeśli ham jest więcej niż spam, możemy zrobić downsampling dla ham
if len(spam) < len(ham):
    ham = resample(ham, replace=False, n_samples=len(spam), random_state=42)
df_balanced = pd.concat([spam, ham])

# Podział na dane treningowe i testowe
X = df_balanced['text']
y = df_balanced['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Wektoryzacja z użyciem TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')  # Używamy stop-words
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Trenowanie modelu
model = MultinomialNB(alpha=0.1)  # Możemy dostosować parametr alpha
model.fit(X_train_vect, y_train)

# Cross-validation (opcjonalne)
cv_scores = cross_val_score(model, X_train_vect, y_train, cv=5)
print(f"Średnia dokładność z cross-validation: {cv_scores.mean()}")

# Zapisz model i wektoryzator do plików
os.makedirs('model', exist_ok=True)  # Tworzymy folder, jeśli nie istnieje
with open('model/spam_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('model/vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

# Oblicz dokładność modelu
predictions = model.predict(X_test_vect)
print("Dokładność:", accuracy_score(y_test, predictions))
