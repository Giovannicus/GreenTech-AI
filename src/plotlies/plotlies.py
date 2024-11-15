import plotly.graph_objects as go
import torch
from typing import List, Optional, Union
import torch
import os
from plotly.subplots import make_subplots
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output

def plot_class_distribution(
    label_counts: torch.Tensor,
    class_names: Optional[Union[List[str], List[int]]] = None,
    title: str = "Class Distribution",
    pull_value: float = 0.02,
    paper_bgcolor: str = None, 
    plot_bgcolor: str = None,
    t: float = 50,
    b: float =50, 
    l: float =20, 
    r: float =20,
    height: float =400,  
    width: float =400,
    legend_x_anchor: str =  "center",
    legend_y_anchor:str = "bottom",
    legend_orientation: str = "h",
    legend_y: str = -.2,
    legend_x: str = .5
) -> go.Figure:
    """
    Create a pie chart visualization of class distribution using Plotly.
    
    Args:
        label_counts (torch.Tensor): Tensor containing the count for each class
        class_names (Optional[Union[List[str], List[int]]]): List of class names from dataset.classes
                                                            or indices if not available
        title (str): Title for the plot
    
    Returns:
        go.Figure: Plotly figure object containing the pie chart
    """

    counts = label_counts.cpu().tolist()
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(counts))]
    
    if len(class_names) != len(counts):
        raise ValueError(
            f"Number of class names ({len(class_names)}) "
            f"doesn't match number of classes in counts ({len(counts)})"
        )
    

    total = sum(counts)
    percentages = [f"{(count/total)*100:.1f}%" for count in counts]
    
    hover_text = [
        f"{name}<br>Count: {int(count)}<br>Percentage: {pct}" 
        for name, count, pct in zip(class_names, counts, percentages)
    ]
    
    fig = go.Figure(data=[
        go.Pie(
            pull=[pull_value] * len(counts),
            labels=class_names,
            values=counts,
            hovertext=hover_text,
            hoverinfo="text",
            textinfo="label+percent",
            hole=0.3,  
            textposition="outside",
            textfont=dict(size=12),
        )
    ])
    
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.95,
            'font': {'size': 24,'color': '#000000'}
        },
        showlegend=True,
        legend={
            'orientation': legend_orientation,
            'yanchor': legend_y_anchor,
            'y': legend_y,
            'xanchor': legend_x_anchor,
            'x': legend_x
        },
        paper_bgcolor=paper_bgcolor,   
        plot_bgcolor=plot_bgcolor,
        margin=dict(t=t, b=b, l=l, r=r),
        height=height, 
        width=width  
    )
    
    return fig

###### Training monitor class ##############
class TrainingMonitor:
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.train_losses = []
        self.val_losses = []
        self.train_f1s = []
        self.val_f1s = []
        self.epochs = []

        plt.style.use('default')

    def update(self, epoch, train_loss, val_loss, train_f1, val_f1):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_f1s.append(train_f1)
        self.val_f1s.append(val_f1)

        clear_output(wait=True)
        self.plot()

    def plot(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        fig.suptitle('Training Progress', fontsize=16, y=1.02)

        ax1.plot(self.epochs, self.train_losses, 'b-o', label='Training Loss',
                markersize=4, linewidth=2, alpha=0.8)
        ax1.plot(self.epochs, self.val_losses, 'r-o', label='Validation Loss',
                markersize=4, linewidth=2, alpha=0.8)
        ax1.set_title('Loss Over Time', pad=10)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_yscale('log')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper right', frameon=True)

        if self.train_losses:
            ax1.annotate(f'{self.train_losses[-1]:.4f}',
                        (self.epochs[-1], self.train_losses[-1]),
                        xytext=(5, 5), textcoords='offset points',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
            ax1.annotate(f'{self.val_losses[-1]:.4f}',
                        (self.epochs[-1], self.val_losses[-1]),
                        xytext=(5, 5), textcoords='offset points',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))


        ax2.plot(self.epochs, self.train_f1s, 'b-o', label='Training F1',
                markersize=4, linewidth=2, alpha=0.8)
        ax2.plot(self.epochs, self.val_f1s, 'r-o', label='Validation F1',
                markersize=4, linewidth=2, alpha=0.8)
        ax2.set_title('F1 Over Time', pad=10)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='lower right', frameon=True)
        ax2.set_ylim(0, 1)

        if self.train_f1s:
            ax2.annotate(f'{self.train_f1s[-1]:.4f}',
                        (self.epochs[-1], self.train_f1s[-1]),
                        xytext=(5, 5), textcoords='offset points',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
            ax2.annotate(f'{self.val_f1s[-1]:.4f}',
                        (self.epochs[-1], self.val_f1s[-1]),
                        xytext=(5, 5), textcoords='offset points',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        plt.tight_layout()
        plt.show()

    def save(self, filepath):
        """Save the current plot to a file"""
        self.plot()
        plt.savefig(filepath, bbox_inches='tight', dpi=300)

    def get_current_metrics(self):
        if not self.epochs:
            return None

        return {
            'epoch': self.epochs[-1],
            'train_loss': self.train_losses[-1],
            'val_loss': self.val_losses[-1],
            'train_f1': self.train_f1s[-1],
            'val_f1': self.val_f1s[-1]
        }

    def stop(self):
        """Cleanup method"""
        plt.close('all')
###### Training monitor class ##############

###### Show exp monitor class ##############
class ShowExpMonitor:
    def __init__(self, train_history_fpath, val_history_fpath):
        self.train_history_fpath = train_history_fpath
        self.val_history_fpath = val_history_fpath

    def create_figure(self):
        """Create and return the plotly figure"""
        # Leggi i dati
        train_df = pd.read_csv(self.train_history_fpath, names=['epoch', 'loss', 'f1'])
        val_df = pd.read_csv(self.val_history_fpath, names=['epoch', 'loss', 'f1'])

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Loss Over Time', 'f1 Over Time'),
            vertical_spacing=0.15
        )

        fig.add_trace(
            go.Scatter(x=train_df['epoch'], y=train_df['loss'],
                      name='Train Loss', mode='lines+markers',
                      line=dict(color='blue')),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=train_df['epoch'], y=train_df['f1'],
                      name='Train f1', mode='lines+markers',
                      line=dict(color='blue')),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=val_df['epoch'], y=val_df['loss'],
                      name='Val Loss', mode='lines+markers',
                      line=dict(color='red')),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=val_df['epoch'], y=val_df['f1'],
                      name='Val f1', mode='lines+markers',
                      line=dict(color='red')),
            row=2, col=1
        )

        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Training Progress",
            title_x=0.5,
            plot_bgcolor='white',  
            paper_bgcolor='white'
        )

        fig.update_xaxes(
            showgrid=True,
            gridcolor='lightgrey',
            gridwidth=0.2,
            griddash='dot',  
            zeroline=False,
            linecolor='black',
            linewidth=1,
            title_text="Epoch"
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor='lightgrey',
            gridwidth=0.2,
            griddash='dot', 
            zeroline=False,
            linecolor='black',
            linewidth=1
        )

        fig.update_yaxes(title_text="Loss", type="log", row=1, col=1)
        fig.update_yaxes(title_text="f1", row=2, col=1)

        return fig

    def plot(self):
        """Display the plot"""
        fig = self.create_figure()
        return fig  

###### Show exp monitor class ##############

def plot_training_history(experiment_dir):
    """Helper function to quickly plot training history"""
    train_path = os.path.join(experiment_dir, "history", "train.csv")
    val_path = os.path.join(experiment_dir, "history", "val.csv")

    monitor = ShowExpMonitor(train_path, val_path)
    return monitor.plot() 

def plot_confusion_matrix(confusion_matrix: torch.Tensor, class_names: list):
    """
    Visualizza la confusion matrix usando Plotly con valori assoluti.

    Args:
        confusion_matrix: torch.Tensor - La matrice di confusione
    """
    
    cm = confusion_matrix.cpu().numpy()

   
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate="%{z}",
        textfont={"size": 14},
        hoverongaps=False,
        hovertemplate="Vero: %{y}<br>Predetto: %{x}<br>Valore: %{z}<extra></extra>"
    ))

    
    fig.update_layout(
        title={
            'text': 'Matrice di Confusione',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        xaxis_title="Classe Predetta",
        yaxis_title="Classe Vera",
        xaxis={'side': 'bottom'},
        width=800,
        height=800,
        yaxis={'autorange': 'reversed'}
    )

    fig.show()