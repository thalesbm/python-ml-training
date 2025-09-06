import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def display_graph(dataset: pd.DataFrame):
    print("display_graph()")

    _column_graph(dataset=dataset)

def _column_graph(dataset: pd.DataFrame):
    print("_column_graph()")

    graph_dataset = pd.concat([
        dataset[['year', 'team_a_code', 'team_a_score']].rename(
            columns={'team_a_code': 'team', 'team_a_score': 'goals'}
        ),
        dataset[['year', 'team_b_code', 'team_b_score']].rename(
            columns={'team_b_code': 'team', 'team_b_score': 'goals'}
        ),
    ], ignore_index=True)

    totals = (graph_dataset
                .groupby('team', as_index=False)['goals']
                .sum()
                .sort_values('goals', ascending=False))

    top_n = 20
    ax = totals.head(top_n).plot(kind='bar', x='team', y='goals', figsize=(12,6), legend=False)
    ax.set_xlabel("Seleção")
    ax.set_ylabel("Total de gols")
    ax.set_title(f"Top {top_n} seleções por gols em Copas do Mundo")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
