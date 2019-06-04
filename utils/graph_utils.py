import torch
import torch.nn.functional as F

import numpy as np


def tour_nodes_to_W(nodes):
    """Helper function to convert ordered list of tour nodes to edge adjacency matrix.
    """
    W = np.zeros((len(nodes), len(nodes)))
    for idx in range(len(nodes) - 1):
        i = int(nodes[idx])
        j = int(nodes[idx + 1])
        W[i][j] = 1
        W[j][i] = 1
    # Add final connection of tour in edge target
    W[j][int(nodes[0])] = 1
    W[int(nodes[0])][j] = 1
    return W


def tour_nodes_to_tour_len(nodes, W_values):
    """Helper function to calculate tour length from ordered list of tour nodes.
    """
    tour_len = 0
    for idx in range(len(nodes) - 1):
        i = nodes[idx]
        j = nodes[idx + 1]
        tour_len += W_values[i][j]
    # Add final connection of tour in edge target
    tour_len += W_values[j][nodes[0]]
    return tour_len


def W_to_tour_len(W, W_values):
    """Helper function to calculate tour length from edge adjacency matrix.
    """
    tour_len = 0
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if W[i][j] == 1:
                tour_len += W_values[i][j]
    tour_len /= 2  # Divide by 2 because adjacency matrices are symmetric
    return tour_len


def is_valid_tour(nodes, num_nodes):
    """Sanity check: tour visits all nodes given.
    """
    return sorted(nodes) == [i for i in range(num_nodes)]


def mean_tour_len_edges(x_edges_values, y_pred_edges):
    """
    Computes mean tour length for given batch prediction as edge adjacency matrices (for PyTorch tensors).

    Args:
        x_edges_values: Edge values (distance) matrix (batch_size, num_nodes, num_nodes)
        y_pred_edges: Edge predictions (batch_size, num_nodes, num_nodes, voc_edges)

    Returns:
        mean_tour_len: Mean tour length over batch
    """
    y = F.softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
    y = y.argmax(dim=3)  # B x V x V
    # Divide by 2 because edges_values is symmetric
    tour_lens = (y.float() * x_edges_values.float()).sum(dim=1).sum(dim=1) / 2
    mean_tour_len = tour_lens.sum().to(dtype=torch.float).item() / tour_lens.numel()
    return mean_tour_len


def mean_tour_len_nodes(x_edges_values, bs_nodes):
    """
    Computes mean tour length for given batch prediction as node ordering after beamsearch (for Pytorch tensors).

    Args:
        x_edges_values: Edge values (distance) matrix (batch_size, num_nodes, num_nodes)
        bs_nodes: Node orderings (batch_size, num_nodes)

    Returns:
        mean_tour_len: Mean tour length over batch
    """
    y = bs_nodes.cpu().numpy()
    W_val = x_edges_values.cpu().numpy()
    running_tour_len = 0
    for batch_idx in range(y.shape[0]):
        for y_idx in range(y[batch_idx].shape[0] - 1):
            i = y[batch_idx][y_idx]
            j = y[batch_idx][y_idx + 1]
            running_tour_len += W_val[batch_idx][i][j]
        running_tour_len += W_val[batch_idx][j][0]  # Add final connection to tour/cycle
    return running_tour_len / y.shape[0]


def get_max_k(dataset, max_iter=1000):
    """
    Given a TSP dataset, compute the maximum value of k for which the k'th nearest neighbor 
    of a node is connected to it in the groundtruth TSP tour.
    
    For each node in all instances, compute the value of k for the next node in the tour, 
    and take the max of all ks.
    """ 
    ks = []
    for _ in range(max_iter):
        batch = next(iter(dataset))
        for idx in range(batch.edges.shape[0]):
            for row in range(dataset.num_nodes):
                # Compute indices of current node's neighbors in the TSP solution
                connections = np.where(batch.edges_target[idx][row]==1)[0]
                # Compute sorted list of indices of nearest neighbors (ascending order)
                sorted_neighbors = np.argsort(batch.edges_values[idx][row], axis=-1)  
                for conn_idx in connections:
                    ks.append(np.where(sorted_neighbors==conn_idx)[0][0])
    # print("Ks array counts: ", np.unique(ks, return_counts=True))
    # print(f"Mean: {np.mean(ks)}, StdDev: {np.std(ks)}")
    return int(np.max(ks))
