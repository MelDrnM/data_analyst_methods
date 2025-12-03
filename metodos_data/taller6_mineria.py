import string
from sklearn.feature_extraction.text import TfidfVectorizer
docs = [
    "Las rosas rojas florecen en primavera con un aroma dulce y embriagador",
    "Los girasoles siguen al sol durante todo el día buscando la luz",
    "Las margaritas blancas crecen silvestres en los campos verdes",
    "Los tulipanes de colores vibrantes adornan los jardines holandeses",
    "Las orquídeas exóticas requieren cuidados especiales para florecer"]

frases_minusculas = [doc.lower() for doc in docs]
frases_limpias = [frase.translate(str.maketrans('', '', string.punctuation)) for frase in frases_minusculas]


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)
print(vectorizer.get_feature_names_out())
print(X.toarray())

