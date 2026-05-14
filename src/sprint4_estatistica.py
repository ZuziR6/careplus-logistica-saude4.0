CÉLULA 1

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Criação de base de dados sintética (Saúde 4.0)
np.random.seed(42)
n_samples = 500

data = {
    'idade': np.random.randint(20, 70, n_samples),
    'imc': np.random.uniform(18, 35, n_samples),
    'horas_sono': np.random.uniform(4, 9, n_samples),
    'bpm_repouso': np.random.randint(50, 110, n_samples),
    'historico_familiar': np.random.randint(0, 2, n_samples)
}

df = pd.DataFrame(data)

# Definindo a Variável Alvo (Binária): Risco de Hipertensão/Cardíaco (1 ou 0)
# Criamos uma lógica onde idade avançada, IMC alto e pouco sono aumentam o risco
prob = (0.03 * df['idade'] + 0.1 * df['imc'] - 0.2 * df['horas_sono'] + 0.02 * df['bpm_repouso'])
df['target_risco'] = (prob > prob.median()).astype(int)

# 2. Definição de Variáveis
X = df.drop('target_risco', axis=1) # Explicativas
y = df['target_risco']              # Alvo

# 3. Divisão Treino/Teste e Escalonamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Etapa 1 concluída: Variáveis definidas e dados normalizados.")


CÉLULA 2 

from sklearn.linear_model import LogisticRegression

# 1. Construção do Modelo (Item b)
modelo = LogisticRegression()
modelo.fit(X_train_scaled, y_train)

# 2. Extração de Coeficientes para a Função (Item c)
intercepto = modelo.intercept_[0]
coeficientes = modelo.coef_[0]
features = X.columns

print(f"Intercepto (beta0): {intercepto:.4f}")
for feature, coef in zip(features, coeficientes):
    print(f"Coeficiente {feature} (beta): {coef:.4f}")

# 3. Exemplo Prático: Cálculo de Probabilidade para Novo Dado (Item d)
# Exemplo: Paciente de 50 anos, IMC 30, 5h de sono, 90 BPM, Histórico Sim(1)
# Versão corrigida para evitar o Warning
novo_paciente = pd.DataFrame([[50, 30, 5, 90, 1]], columns=features)
novo_paciente_scaled = scaler.transform(novo_paciente)

probabilidade = modelo.predict_proba(novo_paciente_scaled)[0][1]
print(f"Probabilidade de Alto Risco: {probabilidade:.2%}")

CÉLULA 3

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score, roc_curve)

# 1. Predições
y_pred = modelo.predict(X_test_scaled)
y_probs = modelo.predict_proba(X_test_scaled)[:, 1]

# 2. Cálculo das Métricas (Item e)
metrics = {
    "Acurácia": accuracy_score(y_test, y_pred),
    "Precisão": precision_score(y_test, y_pred),
    "Recall (Sensibilidade)": recall_score(y_test, y_pred),
    "F1-score": f1_score(y_test, y_pred),
    "ROC AUC": roc_auc_score(y_test, y_probs)
}

print("--- Métricas de Avaliação ---")
for nome, valor in metrics.items():
    print(f"{nome}: {valor:.4f}")

# 3. Visualizações (Itens obrigatórios)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Matriz de Confusão')
ax[0].set_xlabel('Predito')
ax[0].set_ylabel('Real')

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_probs)
ax[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {metrics["ROC AUC"]:.2f})')
ax[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax[1].set_title('Curva ROC')
ax[1].set_xlabel('Taxa de Falsos Positivos')
ax[1].set_ylabel('Taxa de Verdadeiros Positivos')
ax[1].legend(loc="lower right")

plt.tight_layout()
plt.show()

