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

Etapa: Problema           
Ação: Formular tarefa & *target*
Objetivo: Ex.: prever **resultado A/D/B** por partida. Unidade = partida.  

Etapa: Métrica & Baseline 
Escolher métrica e baseline  
**Macro-F1** (equilibra classes) + baseline da classe majoritária

Auditoria
Inspecionar dados
`df.info()`, `isna()`, duplicatas, distribuição de `outcome`. 


Split 
Estratégia de separação 
**GroupSplit por `year`** (ou hold-out por ano mais recente)

| 5. Pré-processamento  | Tipos + imputação + encoding | Categóricas (`stage_name`, `team_a_code`, `team_b_code`) → One-Hot; tratar `NaN`. |


| 6. Categorias raras   | Agrupar rótulos raros        | Reduz esparsidade e overfitting (ex.: mapear times muito raros para `OTHER`). 


    |
| 7. Anti-vazamento     | Regras de features           | Usar **apenas** informações **pré-jogo**. Gols/flags só para criar o *target*.    |


| 8. Baseline           | Modelo simples               | Regressão Logística multinomial para probabilidades iniciais.     

                |
| 9. Validação          | CV por grupos                | `GroupKFold(year)`; reportar ACC e **Macro-F1** por fold.  

                       |
| 10. Tuning            | Hiperparâmetros              | Grid/Optuna; depois **refit** no conjunto de treino completo.  

                   |
| 11. Avaliação final   | Hold-out + calibração        | Matriz de confusão; opcional `CalibratedClassifierCV`.   

                         |
| 12. Produção          | Empacotar pipeline           | `joblib.dump(pipe, ...)`; função `predict_proba` com checagem de entrada.
