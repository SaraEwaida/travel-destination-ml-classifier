\documentclass[11pt,a4paper]{article}
\usepackage[utf-8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{xcolor}

\lstset{
    language=Python,
    basicstyle=\ttfamily\small,
    keywordstyle=\color{blue},
    commentstyle=\color{gray},
    stringstyle=\color{red},
    breaklines=true,
    tabsize=2
}

\title{Travel Destination Classification}
\author{Sara Ewaida \& Yara Obaid}
\date{January 2026}

\begin{document}

\maketitle

\section*{Overview}

Multi-class text classification: predict travel destination countries from descriptions.

\begin{tabular}{ll}
Dataset & 1,011 samples, 54 countries \\
Feature & TF-IDF (1000 features, 1-2 grams) \\
Best Model & Logistic Regression (66.49\% accuracy) \\
Language & Python 3.7+
\end{tabular}

\section*{Results}

\begin{tabular}{lcc}
Model & Accuracy & F1-Score \\
\hline
k-NN & 36.76\% & 0.406 \\
\textbf{Logistic Regression} & \textbf{66.49\%} & \textbf{0.611} \\
Random Forest & 58.38\% & 0.545 \\
\end{tabular}

\section*{Installation}

\begin{lstlisting}
pip install scikit-learn pandas numpy matplotlib seaborn
\end{lstlisting}

\section*{Usage}

\begin{lstlisting}[language=Python]
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
X = vectorizer.fit_transform(descriptions)

model = LogisticRegression(C=10.0)
model.fit(X_train, y_train)
\end{lstlisting}

\section*{Documentation}

\begin{itemize}
    \item \texttt{Project\_Report.pdf} --- Full analysis and results
    \item \texttt{01\_EDA.ipynb} --- Exploratory data analysis
    \item \texttt{02\_preprocessing.ipynb} --- Data cleaning
    \item \texttt{03\_modeling.ipynb} --- Model training
\end{itemize}

\section*{Key Insights}

\begin{itemize}
    \item Country-specific keywords are strong predictors
    \item Class imbalance affects minority classes
    \item Logistic Regression outperforms ensemble methods
    \item Hierarchical classification improves interpretability
\end{itemize}

\section*{Authors}

Sara Ewaida (1203048), Yara Obaid (1212482)

Birzeit University | ENCS5341 | Jan 2026

\end{document}
