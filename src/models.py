"""
Neural nets used for height estimation.
"""
import torch.nn as nn
import torch
import torch.nn.functional as F 


class Activation(nn.Module):
    """
    Interface between torch activation functions and user specifiied
    model configuration.

    Copied from segmentation-models-pytorch.

    Parameters
    ----------
    name: str
        Name of the activation function to use
    """
    def __init__(self, name, **params):
        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif name == "leaky_relu":
            self.activation = nn.LeakyReLU(**params)
        elif name == "relu":
            self.activation = nn.ReLU()
        elif name == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(
                f"Activation should be sigmoid/softmax/logsoftmax/tanh/relu/leakyrelu"
                f"got {name}"
            )

    def forward(self, x):
        return self.activation(x)


class MLPBlock(nn.Module):
    """
    Multilayer perceptron block. A single hidden layer
    either with or without dropout.

    Parameters
    ----------
    in_dim: int
        Input dimension
    out_dim: int
        Output dimension
    dropout_p: float (optional)
        Dropout probability, should be between 0 and 1 
    """
    
    def __init__(self, in_dim, out_dim, dropout_p=None, batch_norm=False, bias=False):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim, bias=bias)
        if dropout_p is not None:
            dropout = nn.Dropout(dropout_p)
            self.layers = nn.ModuleList([linear, dropout])
        else:
            if batch_norm:
                bn = nn.BatchNorm1d(out_dim)
                self.layers = nn.ModuleList([linear, bn])
            else:
                self.layers = nn.ModuleList([linear])
            
    def forward(self, x):
        """
        Apply layer to input
        """
        for layer in self.layers:
            x = layer(x)
        return x
        
    
class RegressionMLP(nn.Module):
    """ 
    Multi-layer perceptron for regression
    
    Parameters
    ----------
    in_dim: int
        Input dimension
    out_dim: int
        Output dimension
    hidden_layer_sizes: list(int)
        Number of units per hidden layer
    activation: str (optional)
        Activation function to use for hidden layers
    dropout_p: float (optional)
        Dropout probability, should be between 0 and 1 
    batch_norm : bool (optional)
        Whether or not to use a batch norm.
    """ 
    def __init__(self, in_dim, out_dim, hidden_layer_sizes, activation="relu", dropout_p=None, batch_norm=False):
        super().__init__()
        
        self.layers = []
        
        # Input dimension of each layer
        in_dims = [in_dim] + hidden_layer_sizes
        # Output dimensions of each layer
        out_dims = hidden_layer_sizes + [out_dim]

        # Set activation function
        self.activation = Activation(activation)

        # Add layers
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):

            # Only add dropout if we're not at the final layer
            if i < len(hidden_layer_sizes):
                self.layers.append(MLPBlock(in_dim, out_dim,
                                             dropout_p=dropout_p,
                                            batch_norm=batch_norm))
            else:
                self.layers.append(MLPBlock(in_dim, out_dim, dropout_p=None,
                 batch_norm=batch_norm))
                
        # Transform to ModuleList
        self.layers = nn.ModuleList(self.layers)

        gain = 1.0 #torch.nn.init.calculate_gain(activation)
        for layer in self.layers:
            size = layer.layers[0].weight.data.shape

            std = gain * (6/(size[0] + size[1]))**0.5

            nn.init.xavier_uniform_(layer.layers[0].weight.data, gain=gain)

            nn.init.uniform_(layer.layers[0].bias.data, a=-std, b=std)

    
    def forward(self, x):
        """
        Apply model to input
        """
        # Loop through all layers but the last one
        for layer in self.layers[:-1]:
            # Fully connected + dropout
            x = layer(x)
            # Activation
            x = self.activation(x)
        
        # Apply final layer (no activation function)
        x = self.layers[-1](x)
        return x