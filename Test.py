import re
from unicodedata import normalize


def preprocess_text(text, NLP):

    # A minusculas
    text = text.lower()

    # Eliminar formatos (/n /t ...)
    rgx_b = r'(\\.)+'
    text = re.sub(rgx_b, ' ', text).strip()

    # Cambiar signos de puntuación por espacios
    rgx_a = r'\s*[^\w\s]+\s*'
    text = re.sub(rgx_a, ' ', text).strip()

    # Tagging & Lemmatización
    text = " ".join([e.lemma_ for e in NLP(text)])

    # Eliminar accentos?
    rgx_c = r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+"
    text = normalize('NFC', re.sub(rgx_c, r"\1", normalize("NFD", text), 0, re.I))

    # Eliminar números?
    rgx_d = r"\s*\d+\s*"
    text = re.sub(rgx_d, ' ', text).strip()

    return text


city = "gijon"

if city in ["gijon", "madrid", "barcelona"]:
    print("-")
    import es_core_news_sm as spacy_es_model  # python -m spacy download es_core_news_sm
    NLP = spacy_es_model.load(disable=["parser", "ner", "attribute_ruler"])

elif city in ["newyorkcity", "london"]:
    print("--")
    import en_core_web_sm as spacy_en_model  # python -m spacy download en_core_web_sm
    NLP = spacy_en_model.load(disable=["parser", "ner", "attribute_ruler"])
else:
    print("---")
    import fr_core_news_sm as spacy_fr_model  # python -m spacy download fr_core_news_sm
    NLP = spacy_fr_model.load(disable=["parser", "ner", "attribute_ruler"])

print(preprocess_text("Donde puedo comer comida vegana o vegetariana", NLP))
print(preprocess_text("Donde puedo comer comida para veganos", NLP))
print(preprocess_text("Donde puedo comer comida para veganas", NLP))
print(preprocess_text("Donde puedo comer comida sin gluten", NLP))
