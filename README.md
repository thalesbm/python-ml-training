# python-ml-training

### Execução

```bash
# Ative o ambiente virtual (opcional)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Execute os testes de validação
python (arquivo desejado)
```

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
