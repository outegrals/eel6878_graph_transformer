from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

def main() -> None:
    dataset = Planetoid(
        root="data/Cora",
        name="Cora",
        transform=NormalizeFeatures()
    )

    data = dataset[0]

    print("Dataset:", dataset)
    print("Number of graphs:", len(dataset))
    print("Number of node features:", dataset.num_node_features)
    print("Number of classes:", dataset.num_classes)
    print("Number of nodes:", data.num_nodes)
    print("Number of edges:", data.num_edges)
    print("Train nodes:", int(data.train_mask.sum()))
    print("Validation nodes:", int(data.val_mask.sum()))
    print("Test nodes:", int(data.test_mask.sum()))
    print("x shape:", data.x.shape)
    print("edge_index shape:", data.edge_index.shape)
    print("y shape:", data.y.shape)

if __name__ == "__main__":
    main()
