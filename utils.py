from torch_geometric.utils import to_dense_adj
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm # for nice bar

def get_attention_scores(model, event):
    model.eval()

    A = to_dense_adj(event.edge_index)[0]
    attention_scores_list = []

    for layer in model.layers:
        h1 = model.apply_embedding(event)
        h, attention_scores = layer.MHGAtt(A, h1)  # Assuming you want to visualize attention scores for Multi-Head Graph Attention
        attention_scores_list.append(attention_scores.detach().numpy())

    num_layers = len(attention_scores_list)
    num_heads = attention_scores_list[0].shape[0]

    # Define node names based on the number of nodes in the graph
    num_nodes = event.num_nodes

    # Calculate the minimum and maximum values of attention scores across all layers and heads
    min_value = min(np.min(scores) for scores in attention_scores_list)
    max_value = max(np.max(scores) for scores in attention_scores_list)

    return attention_scores_list,min_value,max_value, num_nodes, h1, h


def get_graph_for_each_layer(pred):

    # Numero di grafici da creare
    num_graphs = len(pred) // 3

    # Creazione e visualizzazione dei grafici
    for i in range(num_graphs):
        start_idx = i * 3
        end_idx = start_idx + 3
        triplet_pred = pred[start_idx:end_idx]
        
        plt.figure(figsize=(10, 6))
        
        # Traccia i valori
        for idx, y in enumerate(zip(*triplet_pred)):
            x_values = range(len(y))
            plt.plot(x_values, y, label=f'Segnale' if idx == 0 else f'Background')

        # Aggiungi etichette sull'asse x
        plt.xticks(x_values, [f'Layer {i}' for i in range(len(x_values))])
        
        # Aggiungi etichette e titolo
        plt.xlabel('Layer')
        plt.ylabel('Valori')
        plt.title(f'Valori dei layer (Tripletta {i + 1})')
        
        plt.legend()
        plt.grid(True)
        plt.show()
def get_graph_pca(data_list_pca, data_list_class):
    print("Creazione del grafo attraverso le componenti PCA")
    
    # Palette di colori
    colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)', 
              'rgb(148, 103, 189)', 'rgb(140, 86, 75)', 'rgb(227, 119, 194)']
    
    # Creiamo il grafico a dispersione (scatter plot) con Plotly
    fig = go.Figure()

    # Creiamo un dizionario per tracciare quali classi sono già state aggiunte alla leggenda
    class_in_legend = {}

    # Aggiungiamo i punti con le classi e la loro rappresentazione sull'hover
    for index_grafo, data_pca in enumerate(tqdm(data_list_pca)):
        class_label = data_list_class[index_grafo].item()
        if class_label not in class_in_legend:
            show_legend = True
            class_in_legend[class_label] = True
        else:
            show_legend = False
            
        fig.add_trace(go.Scatter(
            x=[data_pca[0]], 
            y=[data_pca[1]], 
            mode='markers', 
            marker=dict(size=5, color=colors[class_label % 7]), 
            name=f'{class_label}',
            text=[f'{class_label}'],  # Testo per l'hover
            hoverinfo='text',  # Mostra il testo sull'hover
            showlegend=show_legend
        ))

    # Impostiamo il layout del grafico
    fig.update_layout(
        title='Scatter plot per ogni nodo',
        xaxis_title='X',
        yaxis_title='Y',
        hovermode='closest',  # Mostra l'hover per il punto più vicino
    )
    # Mostra il grafico
    fig.show()
