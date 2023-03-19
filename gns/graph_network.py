from typing import List
import numpy as np
import dhg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv import HypergraphConv
from torch_scatter import scatter, scatter_max


def build_mlp(
        input_size: int,
        hidden_layer_sizes: List[int],
        output_size: int = None,
        output_activation: nn.Module = nn.Identity,
        activation: nn.Module = nn.ReLU) -> nn.Module:
  """Build a MultiLayer Perceptron.

  Args:
    input_size: Size of input layer.
    layer_sizes: An array of input size for each hidden layer.
    output_size: Size of the output layer.
    output_activation: Activation function for the output layer.
    activation: Activation function for the hidden layers.

  Returns:
    mlp: An MLP sequential container.
  """
  # Size of each layer
  layer_sizes = [input_size] + hidden_layer_sizes
  if output_size:
    layer_sizes.append(output_size)

  # Number of layers
  nlayers = len(layer_sizes) - 1

  # Create a list of activation functions and
  # set the last element to output activation function
  act = [activation for i in range(nlayers)]
  act[-1] = output_activation

  # Create a torch sequential container
  mlp = nn.Sequential()
  for i in range(nlayers):
    mlp.add_module("NN-" + str(i), nn.Linear(layer_sizes[i],
                                             layer_sizes[i + 1]))
    mlp.add_module("Act-" + str(i), act[i]())

  return mlp


class Encoder(nn.Module):
  """Graph network encoder. Encode nodes and edges states to an MLP. The Encode:
  :math: `\mathcal{X} \rightarrow \mathcal{G}` embeds the particle-based state
  representation, :math: `\mathcal{X}`, as a latent graph, :math:
  `G^0 = encoder(\mathcal{X})`, where :math: `G = (V, E, u), v_i \in V`, and
  :math: `e_{i,j} in E`
  """

  def __init__(
          self,
          nnode_in_features: int,
          nnode_out_features: int,
          nedge_in_features: int,
          nedge_out_features: int,
          nmlp_layers: int,
          mlp_hidden_dim: int):
    """The Encoder implements nodes features :math: `\varepsilon_v` and edge
    features :math: `\varepsilon_e` as multilayer perceptrons (MLP) into the
    latent vectors, :math: `v_i` and :math: `e_{i,j}`, of size 128.

    Args:
      nnode_in_features: Number of node input features (for 2D = 30, calculated
        as [10 = 5 times steps * 2 positions (x, y) +
        4 distances to boundaries (top/bottom/left/right) +
        16 particle type embeddings]).
      nnode_out_features: Number of node output features (latent dimension of
        size 128).
      nedge_in_features: Number of edge input features (for 2D = 3, calculated
        as [2 (x, y) relative displacements between 2 particles + distance
        between 2 particles]).
      nedge_out_features: Number of edge output features (latent dimension of
        size 128).
      nmlp_layer: Number of hidden layers in the MLP (typically of size 2).
      mlp_hidden_dim: Size of the hidden layer (latent dimension of size 128).

    """
    super(Encoder, self).__init__()
    # Encode node features as an MLP
    self.node_fn = nn.Sequential(*[build_mlp(nnode_in_features,
                                             [mlp_hidden_dim
                                              for _ in range(nmlp_layers)],
                                             nnode_out_features),
                                   nn.LayerNorm(nnode_out_features)])
    # Encode edge features as an MLP
    self.edge_fn = nn.Sequential(*[build_mlp(nedge_in_features,
                                             [mlp_hidden_dim
                                              for _ in range(nmlp_layers)],
                                             nedge_out_features),
                                   nn.LayerNorm(nedge_out_features)])

  def forward(
          self,
          x: torch.tensor,
          edge_features: torch.tensor):
    """The forward hook runs when the Encoder class is instantiated

    Args:
      x: Particle state representation as a torch tensor with shape
        (nparticles, nnode_input_features)
      edge_features: Edge features as a torch tensor with shape
        (nparticles, nedge_input_features)

    """
    return self.node_fn(x), self.edge_fn(edge_features)


class InteractionNetwork(MessagePassing):
  def __init__(
      self,
      nnode_in: int,
      nnode_out: int,
      nedge_in: int,
      nedge_out: int,
      nmlp_layers: int,
      mlp_hidden_dim: int,
  ):
    """InteractionNetwork derived from torch_geometric MessagePassing class

    Args:
      nnode_in: Number of node inputs (latent dimension of size 128).
      nnode_out: Number of node outputs (latent dimension of size 128).
      nedge_in: Number of edge inputs (latent dimension of size 128).
      nedge_out: Number of edge output features (latent dimension of size 128).
      nmlp_layer: Number of hidden layers in the MLP (typically of size 2).
      mlp_hidden_dim: Size of the hidden layer (latent dimension of size 128).

    """
    # Aggregate features from neighbors
    super(InteractionNetwork, self).__init__(aggr='add')
    # Node MLP
    self.node_fn = nn.Sequential(*[build_mlp(nnode_in + nedge_out,
                                             [mlp_hidden_dim
                                              for _ in range(nmlp_layers)],
                                             nnode_out),
                                   nn.LayerNorm(nnode_out)])
    # Edge MLP
    self.edge_fn = nn.Sequential(*[build_mlp(nnode_in + nnode_in + nedge_in,
                                             [mlp_hidden_dim
                                              for _ in range(nmlp_layers)],
                                             nedge_out),
                                   nn.LayerNorm(nedge_out)])

  def forward(self,
              x: torch.tensor,
              edge_index: torch.tensor,
              edge_features: torch.tensor):
    """The forward hook runs when the InteractionNetwork class is instantiated

    Args:
      x: Particle state representation as a torch tensor with shape
        (nparticles, nnode_input_features)
      edge_index: A torch tensor list of source and target nodes with shape
        (2, nedges)
      edge_features: Edge features as a torch tensor with shape
        (nedges, nedge_in=latent_dim of 128)

    Returns:
      tuple: Updated node and edge features
    """
    # Save particle state and edge features
    x_residual = x
    edge_features_residual = edge_features
    # Start propagating messages.
    # Takes in the edge indices and all additional data which is needed to
    # construct messages and to update node embeddings.
    x, edge_features = self.propagate( #propagate calls -> 
        edge_index=edge_index, x=x, edge_features=edge_features)

    return x + x_residual, edge_features + edge_features_residual

  def message(self,
              x_i: torch.tensor,
              x_j: torch.tensor,
              edge_features: torch.tensor) -> torch.tensor:
    """Constructs message from j to i of edge :math:`e_{i, j}`. Tensors :obj:`x`
    passed to :meth:`propagate` can be mapped to the respective nodes :math:`i`
    and :math:`j` by appending :obj:`_i` or :obj:`_j` to the variable name,
    i.e., :obj:`x_i` and :obj:`x_j`.

    Args:
      x_i: Particle state representation as a torch tensor with shape
        (nparticles, nnode_in=latent_dim of 128) at node i
      x_j: Particle state representation as a torch tensor with shape
        (nparticles, nnode_in=latent_dim of 128) at node j
      edge_features: Edge features as a torch tensor with shape
        (nedges, nedge_in=latent_dim of 128)

    """
    # Concat edge features with a final shape of [nedges, latent_dim*3]
    edge_features = torch.cat([x_i, x_j, edge_features], dim=-1)
    edge_features = self.edge_fn(edge_features)
    return edge_features

  def update(self,
             x_updated: torch.tensor,
             x: torch.tensor,
             edge_features: torch.tensor):
    """Update the particle state representation

    Args:
      x: Particle state representation as a torch tensor with shape 
        (nparticles, nnode_in=latent_dim of 128)
      x_updated: Updated particle state representation as a torch tensor with 
        shape (nparticles, nnode_in=latent_dim of 128)
      edge_features: Edge features as a torch tensor with shape 
        (nedges, nedge_out=latent_dim of 128)

    Returns:
      tuple: Updated node and edge features
    """
    # Concat node features with a final shape of
    # [nparticles, latent_dim (or nnode_in) *2]
    x_updated = torch.cat([x_updated, x], dim=-1)#x_updated comes from summing over edges. 
    x_updated = self.node_fn(x_updated)
    print("x shape: ", x[0,:10])
    print("x updated shape: ", x_updated[0,:10])
    print("edge features shape", edge_features.shape)
    return x_updated, edge_features


class Processor(MessagePassing):
  """The Processor: :math: `\mathcal{G} \rightarrow \mathcal{G}` computes 
  interactions among nodes via :math: `M` steps of learned message-passing, to 
  generate a sequence of updated latent graphs, :math: `G = (G_1 , ..., G_M )`, 
  where :math: `G^{m+1| = GN^{m+1} (G^m )`. It returns the final graph, 
  :math: `G^M = PROCESSOR(G^0)`. Message-passing allows information to 
  propagate and constraints to be respected: the number of message-passing 
  steps required will likely scale with the complexity of the interactions.

  """

  def __init__(
      self,
      nnode_in: int,
      nnode_out: int,
      nedge_in: int,
      nedge_out: int,
      nmessage_passing_steps: int,
      nmlp_layers: int,
      mlp_hidden_dim: int,
  ):
    """Processor derived from torch_geometric MessagePassing class. The 
    processor uses a stack of :math: `M GNs` (where :math: `M` is a 
    hyperparameter) with identical structure, MLPs as internal edge and node 
    update functions, and either shared or unshared parameters. We use GNs 
    without global features or global updates (i.e., an interaction network), 
    and with a residual connections between the input and output latent node 
    and edge attributes.

    Args:
      nnode_in: Number of node inputs (latent dimension of size 128).
      nnode_out: Number of node outputs (latent dimension of size 128).
      nedge_in: Number of edge inputs (latent dimension of size 128).
      nedge_out: Number of edge output features (latent dimension of size 128).
      nmessage_passing_steps: Number of message passing steps.
      nmlp_layer: Number of hidden layers in the MLP (typically of size 2).
      mlp_hidden_dim: Size of the hidden layer (latent dimension of size 128).

    """
    super(Processor, self).__init__(aggr='max')
    # Create a stack of M Graph Networks GNs.
    self.gnn_stacks = nn.ModuleList([
        InteractionNetwork(
            nnode_in=nnode_in,
            nnode_out=nnode_out,
            nedge_in=nedge_in,
            nedge_out=nedge_out,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        ) for _ in range(nmessage_passing_steps)])

  def forward(self,
              x: torch.tensor,
              edge_index: torch.tensor,
              edge_features: torch.tensor):
    """The forward hook runs through GNN stacks when class is instantiated. 

    Args:
      x: Particle state representation as a torch tensor with shape 
        (nparticles, latent_dim)
      edge_index: A torch tensor list of source and target nodes with shape 
        (2, nedges)
      edge_features: Edge features as a torch tensor with shape 
        (nparticles, latent_dim)

    """
    for gnn in self.gnn_stacks:
      x, edge_features = gnn(x, edge_index, edge_features)
    return x, edge_features


class Decoder(nn.Module):
  """The Decoder: :math: `\mathcal{G} \rightarrow \mathcal{Y}` extracts the 
  dynamics information from the nodes of the final latent graph, 
  :math: `y_i = \delta v (v_i^M)`

  """

  def __init__(
          self,
          nnode_in: int,
          nnode_out: int,
          nmlp_layers: int,
          mlp_hidden_dim: int):
    """The Decoder coder's learned function, :math: `\detla v`, is an MLP. 
    After the Decoder, the future position and velocity are updated using an 
    Euler integrator, so the :math: `yi` corresponds to accelerations, 
    :math: `\"{p}_i`, with 2D or 3D dimension, depending on the physical domain.

    Args:
      nnode_in: Number of node inputs (latent dimension of size 128).
      nnode_out: Number of node outputs (particle dimension).
      nmlp_layer: Number of hidden layers in the MLP (typically of size 2).
      mlp_hidden_dim: Size of the hidden layer (latent dimension of size 128).
    """
    super(Decoder, self).__init__()
    self.node_fn = build_mlp(
        nnode_in, [mlp_hidden_dim for _ in range(nmlp_layers)], nnode_out)

  def forward(self,
              x: torch.tensor):
    """The forward hook runs when the Decoder class is instantiated

    Args:
      x: Particle state representation as a torch tensor with shape 
        (nparticles, nnode_in)

    """
    return self.node_fn(x)


class EncodeProcessDecode(nn.Module):
  def __init__(
      self,
      nnode_in_features: int,
      nnode_out_features: int,
      nedge_in_features: int,
      latent_dim: int,
      nmessage_passing_steps: int,
      nmlp_layers: int,
      mlp_hidden_dim: int,
  ):
    """Encode-Process-Decode function approximator for learnable simulator.

    Args:
      nnode_in_features: Number of node input features (for 2D = 30, 
        calculated as [10 = 5 times steps * 2 positions (x, y) + 
        4 distances to boundaries (top/bottom/left/right) + 
        16 particle type embeddings]).
      nnode_out_features:  Number of node outputs (particle dimension).
      nedge_in_features: Number of edge input features (for 2D = 3, 
        calculated as [2 (x, y) relative displacements between 2 particles + 
        distance between 2 particles]).
      latent_dim: Size of latent dimension (128)
      nmlp_layer: Number of hidden layers in the MLP (typically of size 2).
      mlp_hidden_dim: Size of the hidden layer (latent dimension of size 128).

    """
    super(EncodeProcessDecode, self).__init__()
    self._encoder = Encoder(
        nnode_in_features=nnode_in_features,
        nnode_out_features=latent_dim,
        nedge_in_features=nedge_in_features,
        nedge_out_features=latent_dim,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,
    )
    self._processor = Processor(
        nnode_in=latent_dim,
        nnode_out=latent_dim,
        nedge_in=latent_dim,
        nedge_out=latent_dim,
        nmessage_passing_steps=nmessage_passing_steps,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,
    )
    self._simple_hypergcn = SimpleHyperGCN(
        nnode_in = latent_dim,
        nnode_out = latent_dim,
        nedge_in = latent_dim,
        nedge_out = latent_dim,
        nmessage_passing_steps = nmessage_passing_steps,
        hyperconv_hidden_dim = mlp_hidden_dim,
        use_attention = False,
        negative_slope = 0.2
    )
    self._hypergraph_processor = HyperGraphProcessor(
        nnode_in=latent_dim,
        nnode_out=latent_dim,
        nedge_in=latent_dim,
        nedge_out=latent_dim,
        nmessage_passing_steps=nmessage_passing_steps,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,
    )
    self._unignn_processor = UniGNNProcessor(
        nnode_in=latent_dim,
        nnode_out=latent_dim,
        nedge_in=latent_dim,
        nedge_out=latent_dim,
        nmessage_passing_steps=nmessage_passing_steps,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,
    )
    self._decoder = Decoder(
        nnode_in=latent_dim,
        nnode_out=nnode_out_features,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,
    )

  def forward(self,
              x: torch.tensor,
              edge_index: torch.tensor,
              edge_features: torch.tensor):
    """The forward hook runs at instatiation of EncodeProcessorDecode class.

      Args:
        x: Particle state representation as a torch tensor with shape 
          (nparticles, nnode_in_features)
        edge_index: A torch tensor list of source and target nodes with shape 
          (2, nedges)
        edge_features: Edge features as a torch tensor with shape 
          (nedges, nedge_in_features)

    """


    x, edge_features = self._encoder(x, edge_features)

    #x, edge_features = self._processor(x, edge_index, edge_features) # original implementation

    # x, edge_features = self._simple_hypergcn(x = x, hyperedge_index = edge_index, hyperedge_attr = edge_features) # SimpleHyperGCN

    #hg = dhg.Hypergraph(x.shape[0], edge_index) # Build DHG hypergraph
    #x, edge_features = self._hypergraph_processor(x, edge_features, hg) # DHG hypergraph message passing
    vertices, hyperedges = edge_index[0], edge_index[1]
    x, edge_features = self._unignn_processor(x, vertices, hyperedges, edge_features)
    
    x = self._decoder(x)
    return x


class EncodeProcessDecodeBoth(nn.Module):
  #takes in both edges and hyperedges.
  #uses unignn to process hyperedges.

  def __init__(
      self,
      nnode_in_features: int,
      nnode_out_features: int,
      nedge_in_features_g: int,
      nedge_in_features_h: int,
      latent_dim: int,
      nmessage_passing_steps: int,
      nmlp_layers: int,
      mlp_hidden_dim: int,
  ):
    """Encode-Process-Decode function approximator for learnable simulator.

    Args:
      nnode_in_features: Number of node input features (for 2D = 30, 
        calculated as [10 = 5 times steps * 2 positions (x, y) + 
        4 distances to boundaries (top/bottom/left/right) + 
        16 particle type embeddings]).
      nnode_out_features:  Number of node outputs (particle dimension).
      nedge_in_features: Number of edge input features (for 2D = 3, 
        calculated as [2 (x, y) relative displacements between 2 particles + 
        distance between 2 particles]).
      latent_dim: Size of latent dimension (128)
      nmlp_layer: Number of hidden layers in the MLP (typically of size 2).
      mlp_hidden_dim: Size of the hidden layer (latent dimension of size 128).

    """
    super(EncodeProcessDecodeBoth, self).__init__()
    self._encoder_g = Encoder(
        nnode_in_features=nnode_in_features,
        nnode_out_features=latent_dim,
        nedge_in_features=nedge_in_features_g,
        nedge_out_features=latent_dim,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,
    )
    self._encoder_h = Encoder(
        nnode_in_features=nnode_in_features,
        nnode_out_features=latent_dim,
        nedge_in_features=nedge_in_features_h,
        nedge_out_features=latent_dim,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,
    )


    self._processor = Processor(
        nnode_in=latent_dim,
        nnode_out=latent_dim,
        nedge_in=latent_dim,
        nedge_out=latent_dim,
        nmessage_passing_steps=nmessage_passing_steps,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,
    )
    self._simple_hypergcn = SimpleHyperGCN(
        nnode_in = latent_dim,
        nnode_out = latent_dim,
        nedge_in = latent_dim,
        nedge_out = latent_dim,
        nmessage_passing_steps = nmessage_passing_steps,
        hyperconv_hidden_dim = mlp_hidden_dim,
        use_attention = False,
        negative_slope = 0.2
    )
    self._hypergraph_processor = HyperGraphProcessor(
        nnode_in=latent_dim,
        nnode_out=latent_dim,
        nedge_in=latent_dim,
        nedge_out=latent_dim,
        nmessage_passing_steps=nmessage_passing_steps,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,
    )
    self._unignn_processor = UniGNNProcessor(
        nnode_in=latent_dim,
        nnode_out=latent_dim,
        nedge_in=latent_dim,
        nedge_out=latent_dim,
        nmessage_passing_steps=nmessage_passing_steps,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,
    )
    self._unignn_processor_both = UniGNNProcessorBoth(
        nnode_in=latent_dim,
        nnode_out=latent_dim,
        nedge_in=latent_dim,
        nedge_out=latent_dim,
        nmessage_passing_steps=nmessage_passing_steps,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,
    )

    self._decoder = Decoder(
        nnode_in=latent_dim,
        nnode_out=nnode_out_features,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,
    )

  def forward(self,
              x: torch.tensor,
              edge_index_g: torch.tensor,
              edge_features_g: torch.tensor,
              edge_index_h: torch.tensor,
              edge_features_h: torch.tensor):
    """The forward hook runs at instatiation of EncodeProcessorDecode class.

      Args:
        x: Particle state representation as a torch tensor with shape 
          (nparticles, nnode_in_features)
        edge_index: A torch tensor list of source and target nodes with shape 
          (2, nedges)
        edge_features: Edge features as a torch tensor with shape 
          (nedges, nedge_in_features)

    """


    x_g, edge_features_g = self._encoder_g(x, edge_features_g)
    x_h, edge_features_h = self._encoder_h(x, edge_features_h)

    vertices_g, hyperedges_g = edge_index_g[0], edge_index_g[1]
    vertices_h, hyperedges_h = edge_index_h[0], edge_index_h[1]

    x, edge_features_g = self._unignn_processor_both(x_g, vertices_g, hyperedges_g, edge_features_g, x_h, vertices_h, hyperedges_h, edge_features_h)
    
    x = self._decoder(x)
    return x


class SimpleHyperGCN(nn.Module):
  def __init__(
      self,
      nnode_in: int,
      nnode_out: int,
      nedge_in: int,
      nedge_out: int,
      nmessage_passing_steps: int,
      hyperconv_hidden_dim: int,
      use_attention: bool,
      negative_slope: float):
    
    super(SimpleHyperGCN, self).__init__()
    self.hyperconv_stacks = [HypergraphConv(in_channels = nnode_in, out_channels = hyperconv_hidden_dim, use_attention = use_attention, negative_slope = negative_slope)]
    self.hyperconv_stacks += [HypergraphConv(in_channels = hyperconv_hidden_dim, out_channels = hyperconv_hidden_dim, use_attention = use_attention, negative_slope = negative_slope) for _ in range(nmessage_passing_steps - 2)]
    self.hyperconv_stacks += [HypergraphConv(in_channels = hyperconv_hidden_dim, out_channels = nnode_out, use_attention = use_attention, negative_slope = negative_slope)]
    self.hyperconv_stacks = nn.ModuleList(self.hyperconv_stacks)

  def forward(self, x, hyperedge_index, hyperedge_attr):
    for i in range(len(self.hyperconv_stacks) - 1):
      x = self.hyperconv_stacks[i](x, hyperedge_index)
      x = F.relu(x)
    
    x = self.hyperconv_stacks[-1](x, hyperedge_index)

    return x, hyperedge_attr
  

class HyperGraphProcessor(nn.Module):
  def __init__(
      self,
      nnode_in: int,
      nnode_out: int,
      nedge_in: int,
      nedge_out: int,
      nmessage_passing_steps: int,
      nmlp_layers: int,
      mlp_hidden_dim: int,
  ):

    super(HyperGraphProcessor, self).__init__()

    self.hnn_stacks = nn.ModuleList([
        HyperGraphInteractionNetwork(
            nnode_in=nnode_in,
            nnode_out=nnode_out,
            nedge_in=nedge_in,
            nedge_out=nedge_out,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        ) for _ in range(nmessage_passing_steps)])

  def forward(self,
              x: torch.tensor,
              edge_features: torch.tensor,
              hg: dhg.Hypergraph):
    
    for hnn in self.hnn_stacks:
      x, edge_features = hnn(x, edge_features, hg)
    return x, edge_features

    
class HyperGraphInteractionNetwork(nn.Module):
  def __init__(
      self,
      nnode_in: int,
      nnode_out: int,
      nedge_in: int,
      nedge_out: int,
      nmlp_layers: int,
      mlp_hidden_dim: int,
  ):

    super(HyperGraphInteractionNetwork, self).__init__()
    # Node MLP
    self.node_fn = nn.Sequential(*[build_mlp(nnode_in + nedge_out,
                                             [mlp_hidden_dim
                                              for _ in range(nmlp_layers)],
                                             nnode_out),
                                   nn.LayerNorm(nnode_out)])
    # Edge MLP
    self.edge_fn = nn.Sequential(*[build_mlp(nnode_in + nedge_in,
                                             [mlp_hidden_dim
                                              for _ in range(nmlp_layers)],
                                             nedge_out),
                                   nn.LayerNorm(nedge_out)])

  def forward(self,
              x: torch.tensor,
              edge_features: torch.tensor,
              hg: dhg.Hypergraph):

    # Save particle state and edge features
    x_residual = x
    edge_features_residual = edge_features

    # Start propagating messages.
    # Takes in the edge indices and all additional data which is needed to
    # construct messages and to update node embeddings.

    # Message / Send
    aggregated_edge_features = hg.v2e(x, aggr = "sum")
    stacked_edge_features = torch.cat([aggregated_edge_features, edge_features], dim = -1)
    updated_edge_features = self.edge_fn(stacked_edge_features)

    # Aggregate
    aggregated_node_features = hg.e2v(updated_edge_features, aggr = "sum")

    # Update
    stacked_node_features = torch.cat([aggregated_node_features, x], dim = -1)
    updated_node_features = self.node_fn(stacked_node_features)

    return updated_node_features + x_residual, updated_edge_features + edge_features_residual
  

class UniGNNProcessor(nn.Module):
  def __init__(
      self,
      nnode_in: int,
      nnode_out: int,
      nedge_in: int,
      nedge_out: int,
      nmessage_passing_steps: int,
      nmlp_layers: int,
      mlp_hidden_dim: int,
  ):

    super(UniGNNProcessor, self).__init__()

    self.hnn_stacks = nn.ModuleList([
        UniGNNInteractionNetwork(
            nnode_in=nnode_in,
            nnode_out=nnode_out,
            nedge_in=nedge_in,
            nedge_out=nedge_out,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        ) for _ in range(nmessage_passing_steps)])

  def forward(self,
              x: torch.tensor,
              vertices: torch.tensor,
              hyperedges: torch.tensor,
              edge_features: torch.tensor):
    
    for hnn in self.hnn_stacks:
      x, edge_features = hnn(x, vertices, hyperedges, edge_features)
    return x, edge_features


class UniGNNProcessorBoth(nn.Module):
  def __init__(
      self,
      nnode_in: int,
      nnode_out: int,
      nedge_in: int,
      nedge_out: int,
      nmessage_passing_steps: int,
      nmlp_layers: int,
      mlp_hidden_dim: int,
  ):

    super(UniGNNProcessorBoth, self).__init__()

    self.gnn_stacks = nn.ModuleList([
        UniGNNInteractionNetwork(
            nnode_in=nnode_in,
            nnode_out=nnode_out,
            nedge_in=nedge_in,
            nedge_out=nedge_out,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            concatenate=True
        ) for _ in range(nmessage_passing_steps)])
    self.hnn_stacks = nn.ModuleList([
        UniGNNInteractionNetwork(
            nnode_in=nnode_in,
            nnode_out=nnode_out,
            nedge_in=nedge_in,
            nedge_out=nedge_out,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            concatenate=False
        ) for _ in range(nmessage_passing_steps)])


  def forward(self,
              x_g: torch.tensor,
              vertices_g: torch.tensor,
              hyperedges_g: torch.tensor,
              edge_features_g: torch.tensor,
              x_h: torch.tensor,
              vertices_h: torch.tensor,
              hyperedges_h: torch.tensor,
              edge_features_h: torch.tensor):
    #Two ways of doing this. either sum in each layer. Or do them seperately and then sum in the end.
    #I go for the latter approach. Getting the right dimensionality is a nightmare otherwise. feel free to change :)

    #Regular Message passing
    for gnn in self.gnn_stacks:
      x_g, edge_features_g = gnn(x_g, vertices_g, hyperedges_g, edge_features_g)

    #Hypergraph message passing
    for hnn in self.hnn_stacks:
      x_h, edge_features_h = hnn(x_h, vertices_h, hyperedges_h, edge_features_h)

    #unfortunately we cannot add the edges. They will have different nr of dimensions
    return x_h+x_g, edge_features_g



    
# source: https://github.com/jianhao2016/AllSet/blob/main/src/models.py
class UniGNNInteractionNetwork(nn.Module):
  # if concatenate = true, then we are in the 2 uniform setting. We concatenate n1 and n2 information instead of summing.
  def __init__(
      self,
      nnode_in: int,
      nnode_out: int,
      nedge_in: int,
      nedge_out: int,
      nmlp_layers: int,
      mlp_hidden_dim: int,
      concatenate = False
  ):

    super(UniGNNInteractionNetwork, self).__init__()
    # Node MLP
    self.node_fn = nn.Sequential(*[build_mlp(nnode_in + nedge_out,
                                             [mlp_hidden_dim
                                              for _ in range(nmlp_layers)],
                                             nnode_out),
                                   nn.LayerNorm(nnode_out)])
    # Edge MLP 
    # if we concatenate then nnode_in*2, if we do not, then simply nnode_in
    if concatenate:
      self.edge_fn = nn.Sequential(*[build_mlp(nnode_in*2 + nedge_in,#nnode_in*2
                                              [mlp_hidden_dim
                                                for _ in range(nmlp_layers)],
                                              nedge_out),
                                    nn.LayerNorm(nedge_out)])
    else:
      self.edge_fn = nn.Sequential(*[build_mlp(nnode_in + nedge_in,#nnode_in*2
                                        [mlp_hidden_dim
                                          for _ in range(nmlp_layers)],
                                        nedge_out),
                              nn.LayerNorm(nedge_out)])
  
    self.concatenate = concatenate

  def forward(self,
              X: torch.tensor,
              vertices: torch.tensor,
              hyperedges: torch.tensor,
              hyperedge_features: torch.tensor):
    
    # Save particle state and edge features
    X_residual = X
    hyperedge_features_residual = hyperedge_features

    # Start propagating messages.
    # Takes in the edge indices and all additional data which is needed to
    # construct messages and to update node embeddings.

    # Message / Send
    Xve = X[vertices]
    # every other index 
    if self.concatenate:
      idx_send = torch.arange(0,2*hyperedges[-1]+1,2)
      idx_rec = torch.arange(1,2*hyperedges[-1]+1+1,2)
      Xve_send = Xve[idx_send]
      Xve_rec = Xve[idx_rec]
      Xe = torch.cat((Xve_send, Xve_rec), dim=-1)
    else:
      Xe = scatter(Xve, hyperedges, dim = 0, reduce = 'add')

    stacked_hyperedge_features = torch.cat([Xe, hyperedge_features], dim = -1)
    updated_hyperedge_features = self.edge_fn(stacked_hyperedge_features)

    # Aggregate
    Xev = updated_hyperedge_features[hyperedges]
    Xv = scatter(Xev, vertices, dim = 0, reduce = 'add')

    # Update
    stacked_node_features = torch.cat([Xv, X], dim = -1)
    updated_node_features = self.node_fn(stacked_node_features)

    return updated_node_features + X_residual, updated_hyperedge_features + hyperedge_features_residual