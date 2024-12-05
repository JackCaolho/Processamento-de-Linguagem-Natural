# Importando bibliotecas necessárias
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from nltk import tokenize
import nltk
from string import punctuation
import seaborn as sns
import unidecode

# Leitura do dataset
# O dataset contém resenhas de filmes em português e suas classificações (positivo ou negativo)
resenha = pd.read_csv("dados/imdb-reviews-pt-br.csv")

# Explorando alguns exemplos de resenhas
print("Exemplo de resenha negativa:\n")
print(resenha["text_pt"][200])

print("\nExemplo de resenha positiva:\n")
print(resenha["text_pt"][49002])

# Convertendo as classificações de texto para numérico (neg: 0, pos: 1)
resenha["classificacao"] = resenha["sentiment"].replace(["neg", "pos"], [0, 1])

# Exibindo a distribuição das classificações
print("\nDistribuição das classificações:")
print(resenha["classificacao"].value_counts())

# Vetorização de texto usando Bag of Words
# Transformando texto em uma matriz de frequência de palavras
vetorizar = CountVectorizer(lowercase=False, max_features=50)
bag_of_words = vetorizar.fit_transform(resenha["text_pt"])
print("\nDimensão da Bag of Words:", bag_of_words.shape)

# Dividindo os dados em treino e teste
treino, teste, classe_treino, classe_teste = train_test_split(
    bag_of_words, resenha["classificacao"], random_state=42
)

# Treinando um modelo de Regressão Logística
modelo = LogisticRegression(solver="lbfgs")
modelo.fit(treino, classe_treino)

# Avaliando o modelo
previsao_teste = modelo.predict_proba(teste)[:, 1] >= 0.5
acuracia = accuracy_score(classe_teste, previsao_teste.astype(int))
print("\nAcurácia do modelo:", acuracia)

# Visualização de palavras mais frequentes usando WordCloud
def gerar_wordcloud(texto, titulo):
    nuvem_palavras = WordCloud(width=800, height=500, max_font_size=110, collocations=False).generate(texto)
    plt.figure(figsize=(10, 7))
    plt.imshow(nuvem_palavras, interpolation="bilinear")
    plt.axis("off")
    plt.title(titulo)
    plt.show()

# Gerando WordCloud para todas as resenhas
texto_completo = " ".join(resenha["text_pt"])
gerar_wordcloud(texto_completo, "WordCloud - Todas as resenhas")

# Processamento de texto: Remoção de stopwords e pontuações
palavras_irrelevantes = nltk.corpus.stopwords.words("portuguese")
tokenizador = tokenize.WordPunctTokenizer()

def processar_texto(texto):
    palavras_processadas = [
        palavra for palavra in tokenizador.tokenize(texto)
        if palavra not in palavras_irrelevantes and palavra not in punctuation
    ]
    return " ".join(palavras_processadas)

resenha["tratamento_1"] = resenha["text_pt"].apply(processar_texto)

# Gerando WordCloud após tratamento inicial
texto_tratado = " ".join(resenha["tratamento_1"])
gerar_wordcloud(texto_tratado, "WordCloud - Após tratamento inicial")

# Normalização: Removendo acentos
resenha["tratamento_2"] = resenha["tratamento_1"].apply(unidecode.unidecode)

# Lematização/Stemming: Redução de palavras à sua raiz
stemmer = nltk.RSLPStemmer()

def aplicar_stemming(texto):
    palavras_stem = [stemmer.stem(palavra) for palavra in tokenizador.tokenize(texto)]
    return " ".join(palavras_stem)

resenha["tratamento_3"] = resenha["tratamento_2"].apply(aplicar_stemming)

# Avaliando o impacto do tratamento no modelo
def classificar_texto(texto, coluna_texto, coluna_classificacao):
    vetorizador = CountVectorizer(lowercase=False, max_features=50)
    bag_of_words = vetorizador.fit_transform(texto[coluna_texto])
    treino, teste, classe_treino, classe_teste = train_test_split(
        bag_of_words, texto[coluna_classificacao], random_state=42
    )
    modelo = LogisticRegression(solver="lbfgs")
    modelo.fit(treino, classe_treino)
    return modelo.score(teste, classe_teste)

acuracia_tratamento = classificar_texto(resenha, "tratamento_3", "classificacao")
print("\nAcurácia após processamento:", acuracia_tratamento)

# Visualizações adicionais: Distribuição de palavras (Pareto)
def plotar_pareto(texto, coluna_texto, n_palavras):
    todas_palavras = " ".join(texto[coluna_texto])
    frequencia = nltk.FreqDist(tokenizador.tokenize(todas_palavras))
    df_frequencia = pd.DataFrame(frequencia.most_common(n_palavras), columns=["Palavra", "Frequência"])
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df_frequencia, x="Palavra", y="Frequência", color="gray")
    plt.title("Top palavras mais frequentes")
    plt.show()

plotar_pareto(resenha, "tratamento_3", 10)
