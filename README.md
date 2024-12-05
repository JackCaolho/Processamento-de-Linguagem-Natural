Processamento de Linguagem Natural com Python
Este projeto é uma introdução prática ao Processamento de Linguagem Natural (PLN) utilizando Python e bibliotecas populares. O objetivo é explorar técnicas para processar, analisar e classificar textos, além de criar visualizações como nuvens de palavras e gráficos de frequência. O projeto é dividido em duas partes principais, cada uma cobrindo aspectos fundamentais do PLN.

Estrutura do Projeto
Parte 1: Explorando e Vetorizando Dados Textuais
Carga e visualização inicial dos dados:

Os dados são carregados a partir de um arquivo CSV contendo resenhas de filmes, classificadas como positivas ou negativas.
Visualização de exemplos de resenhas e análise do balanceamento de classes.
Vetorização com CountVectorizer:

Demonstração do conceito de bag-of-words.
Conversão de textos para representações numéricas para classificação.
Divisão de dados e treinamento:

Divisão dos dados em conjuntos de treino e teste.
Treinamento de um modelo de regressão logística para classificação de sentimentos.
Visualização com WordCloud:

Geração de nuvens de palavras para análises qualitativas.
Comparação entre resenhas positivas e negativas.
Tokenização:

Segmentação de frases em palavras utilizando nltk.tokenize.
Parte 2: Otimizando o Pré-processamento e a Análise
Refinamento de tokens e remoção de stopwords:

Remoção de palavras irrelevantes (stopwords) e pontuação para melhoria da análise.
Normalização de textos:

Remoção de acentos e conversão para caixa baixa.
Funções para pipeline:

Criação de funções reutilizáveis para visualizações (nuvens de palavras, gráficos de Pareto) e classificação.
Avaliação de modelos:

Teste e comparação de modelos utilizando textos pré-processados.
Dependências
Linguagem: Python 3.x
Bibliotecas:
pandas
nltk
scikit-learn
wordcloud
matplotlib
seaborn
unidecode
Instale as dependências com:

bash
Copiar código
pip install pandas nltk scikit-learn wordcloud matplotlib seaborn unidecode
Como Executar
Baixe os dados:

Certifique-se de que o arquivo imdb-reviews-pt-br.csv está na pasta dados/.
Execute o script:

Utilize um ambiente como Jupyter Notebook ou qualquer IDE que suporte execução de células.
Resultados esperados:

Nuvens de palavras para insights visuais.
Gráficos de frequência para análise de Pareto.
Precisão dos modelos de classificação de sentimentos.
Contribuições
Sinta-se à vontade para:

Reportar problemas.
Melhorar o pipeline de pré-processamento.
Sugerir novos modelos ou abordagens para classificação de sentimentos.
