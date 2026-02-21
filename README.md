\documentclass[11pt,a4paper]{article}
\usepackage[utf-8]{inputenc}
\usepackage[margin=0.8in]{geometry}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{fancyhdr}

\lstset{
    language=Python,
    basicstyle=\ttfamily\small,
    keywordstyle=\color{blue},
    commentstyle=\color{gray},
    stringstyle=\color{red},
    breaklines=true,
    showstringspaces=false,
    tabsize=2,
    backgroundcolor=\color{lightgray!10}
}

\pagestyle{fancy}
\fancyhf{}
\rhead{ENCS5341}
\lhead{Travel Destination Classification}
\cfoot{\thepage}

\title{\textbf{Travel Destination Classification}\\
       \small Multi-Class Text Classification with TF-IDF}
\author{Sara Ewaida (1203048) \& Yara Obaid (1212482)\\
        Birzeit University | Dr. Yazan Abu Farha | Jan 2026}
\date{}

\begin{document}

\maketitle

\section*{Quick Overview}

Multi-class text classification predicting travel destination countries from descriptions. Dataset: 1,011 samples, 54 countries. Best model: Logistic Regression (66.49\% accuracy).

\section*{Dataset}

\begin{tabular}{ll}
Total samples & 1,011 (after cleaning) \\
Training/Test & 740 / 185 \\
Classes & 54 countries \\
Feature & TF-IDF from Description \\
\end{tabular}

\section*{Models \& Results}

\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{Accuracy} & \textbf{F1-Score} & \textbf{Best Param} \\
\midrule
k-NN (baseline) & 36.76\% & 0.4056 & k=1 \\
Logistic Regression & \textbf{66.49\%} & \textbf{0.6106} & C=10.0 \\
Random Forest & 58.38\% & 0.5454 & n\_est=100 \\
Hierarchical & 66.29\% & 0.6115 & Region→Country \\
\bottomrule
\end{tabular}

\section*{Installation}

\begin{lstlisting}
pip install scikit-learn pandas numpy matplotlib seaborn
\end{lstlisting}

\section*{Quick Usage}

\begin{lstlisting}[language=Python]
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# TF-IDF features
vectorizer = TfidfVectorizer(max_features=1000, 
    ngram_range=(1,2), stop_words='english')
X = vectorizer.fit_transform(descriptions)

# Best model
clf = LogisticRegression(C=10.0, max_iter=1000)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
\end{lstlisting}

\section*{Key Findings}

\textbf{What works:} Logistic Regression excels with TF-IDF features. Country-specific keywords (``Venice''→Italy, ``Eiffel''→France) are strong predictors. Hierarchical approach improves interpretability.

\textbf{What doesn't:} k-NN poor performance (curse of dimensionality). Class imbalance hurts rare countries (USA: 40\% acc). Random Forest underperforms linear models.

\section*{Error Analysis}

Top misclassification pairs:
\begin{enumerate}
\item USA → Switzerland (2 errors)
\item Turkey → Italy (2 errors)
\item USA → Italy (2 errors)
\end{enumerate}

Easiest: Spain, Switzerland, Palestine (100\%). Hardest: USA (40\%).

Accuracy by description length: Short \& Medium (\textasciitilde 67\%), Long (\textasciitilde 64\%).

\section*{Limitations \& Future Work}

\textbf{Limitations:} Class imbalance, TF-IDF ignores semantics, missing values.

\textbf{Future:} Word embeddings (Word2Vec, FastText), Transformers (BERT), class weighting, SMOTE resampling.

\section*{Project Files}

See included report and notebooks:
\begin{itemize}
\item \texttt{Project\_Report.pdf} --- Full analysis
\item \texttt{01\_EDA.ipynb} --- Data exploration
\item \texttt{02\_preprocessing.ipynb} --- Data cleaning
\item \texttt{03\_modeling.ipynb} --- Model training
\end{itemize}

\end{document}
