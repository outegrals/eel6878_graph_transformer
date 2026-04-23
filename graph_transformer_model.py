import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import TransformerConv


class GraphTransformer(nn.Module):
    """
    Graph Transformer for node classification using multi-head self-attention.

    Each TransformerConv layer applies multi-head attention over neighboring
    nodes, allowing the model to capture both local structure and, through
    stacking layers, longer-range dependencies — in contrast to the purely
    local aggregation used by GCN and GAT.

    Architecture:
        Layer 1: TransformerConv(in -> hidden, heads) + ELU + Dropout
        Layer 2: TransformerConv(hidden*heads -> out, 1 head)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.6,
        beta: bool = True,
    ) -> None:
        """
        Args:
            in_channels:     Number of input node features.
            hidden_channels: Hidden dimension per attention head.
            out_channels:    Number of output classes.
            heads:           Number of attention heads in the first layer.
            dropout:         Dropout probability applied to features and
                             attention weights.
            beta:            If True, enables the skip-connection gating
                             (beta parameter) introduced in the Transformer
                             for Graphs paper. Helps stabilise training.
        """
        super().__init__()

        self.dropout = dropout

        # Layer 1 — multi-head attention; concatenates head outputs
        self.conv1 = TransformerConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            beta=beta,
            concat=True,   # output dim = hidden_channels * heads
        )

        # Layer 2 — single head; averages to produce class logits
        self.conv2 = TransformerConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            dropout=dropout,
            beta=beta,
            concat=False,  # output dim = out_channels
        )

        # Optional linear skip connection to project input features to
        # the hidden dimension for residual addition after layer 1.
        # Not strictly required but can help on deeper variants.
        self.lin = nn.Linear(in_channels, hidden_channels * heads)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          Node feature matrix  [N, in_channels]
            edge_index: Graph connectivity    [2, E]

        Returns:
            Log-softmax class logits          [N, out_channels]
        """
        # Input dropout (same convention as the original GAT paper)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 1 with ELU activation
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2 — produces raw logits
        x = self.conv2(x, edge_index)
        return x

    def get_attention_weights(self, x: torch.Tensor, edge_index: torch.Tensor):
        self.eval()
        with torch.no_grad():
            x = F.dropout(x, p=self.dropout, training=False)
            _, attn = self.conv1(
                x,
                edge_index,
                return_attention_weights=True,
            )
        return attn