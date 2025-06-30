# 🌸 Classificação de Espécies do Iris Dataset com SVM e Naive Bayes

Este projeto de Iniciação Científica tem como foco a aplicação e comparação de algoritmos de aprendizado supervisionado no **Iris Dataset**, com ênfase em métodos de pré-processamento, engenharia de atributos e avaliação de modelos.

## 🧠 Objetivo

Investigar a eficiência dos classificadores **Support Vector Machine (SVM)** e **Naive Bayes** (GaussianNB e BernoulliNB), utilizando métricas quantitativas e experimentos em Python para identificar qual abordagem oferece maior desempenho e robustez para conjuntos de dados reais.

## ⚙️ Tecnologias Utilizadas

- Python 3.11.7  
- Jupyter Notebook  
- pandas, numpy  
- matplotlib, seaborn  
- scikit-learn  

## 📁 Estrutura do Projeto

```INICIACAO_CIENTIFICA_ML_CLASSIFICAO/
│
├── artigo/                      # Versões dos artigosd
│   └── Modelo Overleaf.pdf
├── data/                      # Conjunto de dados processado
│   └── iris_processado.pkl
│
├── notebooks/                 # Notebooks Jupyter para análises individuais
│   ├── somente_teste.ipynb
│   └── validacao_teste.ipynb
│
├── scripts/                   # Scripts reutilizáveis
│   ├── index.py
│   ├── pre_processing.py
│   └── treino_teste.py
│
├── venv/                      # Ambiente virtual Python (fora do controle de versão)
│
├── README.md                  # Documentação do projeto
├── requirements.txt           # Lista de dependências do projeto
└── .gitignore                 # Arquivos e pastas ignoradas pelo Git
```



## 🧪 Metodologia

- **Pré-processamento de Dados**:
  - Remoção de outliers (IQR)
  - Geração de novas features (áreas, proporções)
  - Normalização (`StandardScaler`)

- **Modelagem**:
  - Comparação entre DummyClassifier, SVM (4 kernels) e Naive Bayes (Gaussian e Bernoulli)
  - Ajuste de hiperparâmetros via `GridSearchCV`
  - Avaliação com validação cruzada (5 folds)

- **Métrica Principal**:
  - Acurácia média ± desvio padrão

## 📊 Resultados

| Modelo                | Acurácia Validação | Acurácia Teste | Observações                             |
|----------------------|--------------------|----------------|-----------------------------------------|
| DummyClassifier      | 0.35               | 0.26           | Baseline (classe majoritária)           |
| SVM (linear)         | 0.96 ± 0.04        | 0.96           | Melhor desempenho e robustez            |
| GaussianNB (ajustado)| 1.00               | 1.00           | Possível overfitting (dados pequenos)   |
| BernoulliNB          | 0.82 ± 0.05        | -              | Inferior, sensível a distribuição binária |

## 💡 Conclusões

- O `SVM` com kernel linear demonstrou ser o modelo mais **consistente e confiável**.
- O `GaussianNB` otimizado alcançou alta acurácia, mas pode estar sofrendo de overfitting devido ao tamanho reduzido do conjunto de dados.
- O pré-processamento teve papel fundamental na performance dos modelos.


## Ambiente Virtual
Ative o ambiente virtual (ou crie um com venv) e instale as dependências:
```
pip install -r requirements.txt
```

## ✨ Autor
Bruno Dezorzi
Estudante de ADS, apaixonado por Ciência de Dados, Astrofísica e IA.
Explorador de padrões, movido por curiosidade e guiado por propósito.

"Sic Parvis Magna" — Sir Francis Drake.