# python-ml-training

### Execução

```bash
# Ative o ambiente virtual (opcional)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Execute os testes de validação
python src/salary/main.py
python src/world_cup/main.py
```

| Técnica                    | Tipo de feature         | Serve pra quê                       | Usar quando…                                          |
| -------------------------- | ----------------------- | ----------------------------------- | ----------------------------------------------------- |
| **StandardScaler**         | Numéricas contínuas     | Normalizar escala                   | Modelos baseados em distância/gradiente               |
| **One-Hot Encoding (OHE)** | Categóricas nominais    | Evitar ordens artificiais           | Modelos lineares/SVM/NN                               |
| **Logistic + OHE**         | Numéricas + categóricas | Classificação linear, interpretável | Baseline, explicabilidade, datasets tabulares simples |

Tipos de Dados:
**Variáveis numéricas contínuas:** Contagem inteiras (número de filhos, número de compras)
**Variaveis numéricas discretras:** Categorias COM ordem natural (nivel: junior > pleno > senior)
**Variaveis categóricas nominais:** Categorias SEM ordem natural (comida: salada, pão, molho)
**Variaveis categóricas ordinais:** Categorias COM ordem natural (nivel: junior > pleno > senior)
**Alta cardinalidade:** Muitas categorias unicas, fica impossivel de categorizar (CEP, ID Produto)
**Binária:** Sexo (0 / 1)


### TO DO

1. Entenda os tipos das colunas → 2. Entenda o tipo de alvo → 3. Escolha família de modelos → 4. Compare com baseline.

Entender o objetivo

**Etapa:** Problema           
**Ação:** Formular tarefa & *target*
**Objetivo:** Ex.: prever **resultado A/D/B** por partida. Unidade = partida.  

**Etapa:** Métrica & Baseline 
**Ação:** Escolher métrica e baseline  
**Objetivo:** **Macro-F1** (equilibra classes) + baseline da classe majoritária

**Etapa:** Auditoria
**Ação:**Inspecionar dados
**Objetivo:** `df.info()`, `isna()`, duplicatas, distribuição de `outcome`. 

**Etapa:** Split 
**Ação:** Estratégia de separação 
**Objetivo:** **GroupSplit por `year`** (ou hold-out por ano mais recente)

**Etapa:** Pré-processamento
**Ação:** Tipos + imputação + encoding
**Objetivo:** Categóricas (`stage_name`, `team_a_code`, `team_b_code`) → One-Hot; tratar `NaN`.

**Etapa:** **Etapa:** Categorias raras
**Ação:** Agrupar rótulos raros
**Objetivo:** Reduz esparsidade e overfitting (ex.: mapear times muito raros para `OTHER`). 
 
**Etapa:** Anti-vazamento
**Ação:** Regras de features
**Objetivo:** Usar **apenas** informações **pré-jogo**. Gols/flags só para criar o *target*.

**Etapa:** Baseline
**Ação:** Modelo simples
**Objetivo:** Regressão Logística multinomial para probabilidades iniciais.     

**Etapa:** Validação
**Ação:** CV por grupos
**Objetivo:** `GroupKFold(year)`; reportar ACC e **Macro-F1** por fold.  

**Etapa:** Tuning
**Ação:** Hiperparâmetros
**Objetivo:** Grid/Optuna; depois **refit** no conjunto de treino completo.  

**Etapa:** Avaliação final
**Ação:** Hold-out + calibração
**Objetivo:** Matriz de confusão; opcional `CalibratedClassifierCV`.   

**Etapa:** Produção
**Ação:** Empacotar pipeline
**Objetivo:** `joblib.dump(pipe, ...)`; função `predict_proba` com checagem de entrada.
