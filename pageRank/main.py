import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Загрузка необходимых ресурсов из NLTK
nltk.download('punkt')  # используется для токенизации
nltk.download('stopwords')

filename = "source.txt"
output_filename = "output.txt"
stop_words = stopwords.words('russian')

# Функция для преобразования текста в векторы
def sentence_to_vec(sents):
    vectorizer = CountVectorizer(stop_words=stop_words).fit(sents)
    return vectorizer.transform(sents).toarray()

# Text Rank алгоритм
def textrank(text, num_sentences=3):
    sentences = sent_tokenize(text)
    sent_vecs = sentence_to_vec(sentences)
    similarity_matrix = cosine_similarity(sent_vecs)
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted([(scores[i], s) for i, s in enumerate(sentences)], reverse=True)
    summary = ' '.join([s for _, s in ranked_sentences[:num_sentences]])
    return summary


# Загрузка текста из файла
with open(filename, 'r', encoding='utf-8') as file:
    text = file.read()

# Генерация краткой сводки текста
summary = textrank(text, 15)

# Сохранение краткой сводки в файл
with open(output_filename, 'w', encoding='utf-8') as file:
    file.write(summary)
