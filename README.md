# Introdução ao Processamento de Linguagem Natural com Python

Este repositório contém exemplos práticos de como trabalhar com **Processamento de Linguagem Natural (PLN)** utilizando Python, abrangendo desde a exploração de dados textuais até a construção de modelos de aprendizado de máquina.

## Estrutura do Projeto

### Parte 1: Explorando e Vetorizando Dados Textuais
- **Objetivo**: Introduzir conceitos básicos de PLN, como a vetorização de textos utilizando `CountVectorizer`.
- **Passos principais**:
  - Carregar e explorar um dataset de resenhas de filmes.
  - Transformar dados textuais em representações vetoriais (Bag of Words).
  - Construir um modelo de Regressão Logística para classificar sentimentos.

### Parte 2: Visualizações e Pré-Processamento de Dados Textuais
- **Objetivo**: Explorar visualizações e técnicas de tokenização.
- **Passos principais**:
  - Gerar nuvens de palavras para sentimentos positivos e negativos.
  - Tokenizar textos e identificar palavras mais frequentes.
  - Implementar pipelines para limpar e processar dados textuais.

### Parte 3: Normalização e Otimização
- **Objetivo**: Aprimorar a análise textual através de técnicas de normalização.
- **Passos principais**:
  - Remoção de acentos e pontuações.
  - Criação de representações textuais otimizadas para modelos.
  - Comparação de resultados entre diferentes níveis de tratamento.

## Requisitos

### Dependências
Certifique-se de ter as seguintes bibliotecas instaladas:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `nltk`
- `wordcloud`
- `unidecode`

### Dataset
O projeto utiliza um dataset de resenhas de filmes, disponível no diretório `dados/`:
- `imdb-reviews-pt-br.csv`: Contém colunas com texto das resenhas (`text_pt`) e sentimentos (`sentiment`).

## Principais Visualizações

### Nuvens de Palavras
Geramos nuvens de palavras para identificar os termos mais frequentes em textos classificados como:
- **Positivos**
- **Negativos**

### Gráficos de Pareto
Barplot das palavras mais frequentes, com possibilidade de customizar o número de palavras exibidas.

## Como Usar

1. Clone este repositório:
   ```bash
   git clone https://github.com/JackCaolho/Processamento-de-Linguagem-Natural.git
