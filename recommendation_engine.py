import pandas as pd
import os
import requests
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'book_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'tfidf_model.joblib')

POLISH_STOPWORDS = [
    "i", "oraz", "a", "w", "z", "na", "do", "nie", "to", "jest",
    "się", "że", "o", "jak", "ale", "albo", "czy", "być", "ponieważ"
]

def prepare_data():
    df = pd.read_csv(
        DATA_PATH,
        dtype={'ISBN': str},
        on_bad_lines='skip',
        engine='python'
    )

    # Upewnij się, że wszystkie wymagane kolumny są obecne
    required_columns = ['ISBN', 'Title', 'Author', 'Category', 'Description']
    for col in required_columns:
        if col not in df.columns:
            df[col] = ''

    # Czyszczenie i normalizacja danych
    df = df[required_columns].drop_duplicates(subset=['ISBN'])
    df['Title'] = df['Title'].astype(str).str.strip().str.lower()
    df['Category'] = df['Category'].fillna('').astype(str).str.replace(';', ' ')
    df['Author'] = df['Author'].fillna('').astype(str)
    df['Description'] = df['Description'].fillna('').astype(str).apply(clean_description)
    df['ISBN'] = df['ISBN'].astype(str).str.strip()

    # Łączenie cech
    df['combined_features'] = (
        df['Title'] + ' ' +
        df['Author'] + ' ' +
        df['Category'] + ' ' +
        df['Description']
    )

    return df.reset_index(drop=True)

def clean_description(text):
    return text.replace('\n', ' ').replace('\r', '').strip() if isinstance(text, str) else ""

def train_tfidf_model(df):
    tfidf = TfidfVectorizer(
        stop_words=POLISH_STOPWORDS,
        ngram_range=(1, 2),
        max_features=5000
    )
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    joblib.dump((tfidf, tfidf_matrix), MODEL_PATH)
    return tfidf, tfidf_matrix

def load_resources():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Brak pliku z danymi!")

    df = prepare_data()
    if df.empty:
        raise ValueError("Brak danych w pliku CSV!")

    # Indeks: tytuły jako indeksy (normalizowane)
    indices = pd.Series(df.index, index=df['Title']).drop_duplicates()

    if not os.path.exists(MODEL_PATH):
        tfidf, tfidf_matrix = train_tfidf_model(df)
    else:
        tfidf, tfidf_matrix = joblib.load(MODEL_PATH)
        if tfidf_matrix.shape[0] != len(df):
            tfidf, tfidf_matrix = train_tfidf_model(df)

    return df, tfidf, indices, tfidf_matrix

def get_recommendations(title, df, indices, tfidf_matrix, n=5):
    title_clean = title.strip().lower()
    if title_clean not in indices.index:
        return pd.DataFrame()

    idx = indices[title_clean]
    if idx >= tfidf_matrix.shape[0]:
        raise ValueError(f"Indeks {idx} wykracza poza macierz TF-IDF")

    cosine_sim = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
    sim_scores = sorted(enumerate(cosine_sim), key=lambda x: x[1], reverse=True)[1:n+1]

    return df.iloc[[i[0] for i in sim_scores]][['Title', 'Author', 'Category', 'Description', 'ISBN']]

def get_bn_metadata(isbn):
    url = f"https://data.bn.org.pl/api/bibs.json?isbn={isbn}"
    try:
        response = requests.get(url, timeout=10)
        return response.json().get('bibs', [{}])[0] if response.status_code == 200 else None
    except Exception as e:
        print(f"Błąd API BN: {e}")
        return None
