import torch.nn as nn

def get_net(in_dim, n_labels, num_layers, dropout=0.0):
	# Copied from modeling.train
    dropouts = [dropout] * (num_layers - 1) if isinstance(dropout, (int, float)) else dropout
    if len(dropouts) + 1 != num_layers:
        raise ValueError("length of dropout must be equal to num_layers - 1")
    net = nn.Sequential()
    for i in range(1, num_layers):
        neurons = max(128 // i, n_labels)
        net.add_module(f"fc{i}", nn.Linear(in_dim, neurons))
        net.add_module(f"relu{i}", nn.ReLU())
        net.add_module(f"dropout{i}", nn.Dropout(p=dropouts[i - 1]))
        in_dim = neurons
    net.add_module("fc_out", nn.Linear(neurons, n_labels))
    # no softmax here since we're using CE loss which includes it
    # net.add_module(Softmax(dim=1))
    return net
