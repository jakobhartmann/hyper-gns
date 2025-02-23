import torch
import torch.nn as nn
import numpy as np
from gns import graph_network
from torch_geometric.nn import MessagePassing, radius_graph, knn_graph
from sklearn.cluster import KMeans
from torch_scatter import scatter_mean, scatter_std, scatter_max, scatter_min
from typing import Dict
import itertools


class LearnedSimulator(nn.Module):
  """Learned simulator from https://arxiv.org/pdf/2002.09405.pdf."""

  def __init__(
          self,
          particle_dimensions: int,
          nnode_in: int,
          nedge_in: int,#if Both graph and hypergraph then nedge_in is graph features
          latent_dim: int,
          nmessage_passing_steps: int,
          nmlp_layers: int,
          mlp_hidden_dim: int,
          connectivity_radius: float,
          boundaries: np.ndarray,
          normalization_stats: Dict,
          nparticle_types: int,
          particle_type_embedding_size,
          device="cpu",
          nedge_in_h = None,
          myflags=None#Only used if both = True. Otherwise features stored in nedge_in.
          ):
    """Initializes the model.

    Args:
      particle_dimensions: Dimensionality of the problem.
      nnode_in: Number of node inputs.
      nedge_in: Number of edge inputs.
      latent_dim: Size of latent dimension (128)
      nmessage_passing_steps: Number of message passing steps.
      nmlp_layers: Number of hidden layers in the MLP (typically of size 2).
      connectivity_radius: Scalar with the radius of connectivity.
      boundaries: Array of 2-tuples, containing the lower and upper boundaries
        of the cuboid containing the particles along each dimensions, matching
        the dimensionality of the problem.
      normalization_stats: Dictionary with statistics with keys "acceleration"
        and "velocity", containing a named tuple for each with mean and std
        fields, matching the dimensionality of the problem.
      nparticle_types: Number of different particle types.
      particle_type_embedding_size: Embedding size for the particle type.
      device: Runtime device (cuda or cpu).

    """
    super(LearnedSimulator, self).__init__()
    self._boundaries = boundaries
    self._connectivity_radius = connectivity_radius
    self._normalization_stats = normalization_stats
    self._nparticle_types = nparticle_types
    self.myflags = myflags
    # Particle type embedding has shape (9, 16)
    self._particle_type_embedding = nn.Embedding(
        nparticle_types, particle_type_embedding_size)

    # Initialize the EncodeProcessDecode
    self._encode_process_decode = graph_network.EncodeProcessDecode(
        nnode_in_features=nnode_in,
        nnode_out_features=particle_dimensions,
        nedge_in_features=nedge_in,
        latent_dim=latent_dim,
        nmessage_passing_steps=nmessage_passing_steps,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim)
    if myflags["USE_BOTH"]:
      self._encode_process_decode_both = graph_network.EncodeProcessDecodeBoth(
          nnode_in_features=nnode_in,
          nnode_out_features=particle_dimensions,
          nedge_in_features_g=nedge_in,#edge in features for graph
          nedge_in_features_h=nedge_in_h,#edge in features for hypergraph
          latent_dim=latent_dim,
          nmessage_passing_steps=nmessage_passing_steps,
          nmlp_layers=nmlp_layers,
          mlp_hidden_dim=mlp_hidden_dim)

    self._device = device

  def forward(self):
    """Forward hook runs on class instantiation"""
    pass

  def _compute_graph_connectivity(
          self,
          node_features: torch.tensor,
          nparticles_per_example: torch.tensor,
          radius: float,
          add_self_edges: bool = True):
    """Generate graph edges to all particles within a threshold radius

    Args:
      node_features: Node features with shape (nparticles, dim).
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      radius: Threshold to construct edges to all particles within the radius.
      add_self_edges: Boolean flag to include self edge (default: True)
    """
    # Specify examples id for particles
    batch_ids = torch.cat(
        [torch.LongTensor([i for _ in range(n)])
         for i, n in enumerate(nparticles_per_example)]).to(self._device)

    # radius_graph accepts r < radius not r <= radius
    # A torch tensor list of source and target nodes with shape (2, nedges)
    edge_index = radius_graph(
        node_features, r=radius, batch=batch_ids, loop=add_self_edges)

    # The flow direction when using in combination with message passing is
    # "source_to_target"
    receivers = edge_index[0, :]
    senders = edge_index[1, :]

    return receivers, senders

  def _compute_graph_connectivity_hypergraph(
          self,
          node_features: torch.tensor,
          nparticles_per_example: torch.tensor,
          radius: float,
          add_self_edges: bool = True):
        # returns 2xk hyperedge matrix, nr_edges. Does it based on KNN and KMeans clustering as described in the paper.

        batch_ids = torch.cat([torch.LongTensor([i for _ in range(n)]) for i, n in enumerate(nparticles_per_example)]).to(self._device)
        top_s = self.myflags["top_s"]
        nr_cl = self.myflags["k_m_cl"]  #nr clusters
        knn   = self.myflags["k_nn_nr"] #nr k nearest neighbours
        radius= self.myflags["con_rad"]
        if self.myflags["knn_clustering"]:
          edge_index = knn_graph(node_features, k=knn, batch=batch_ids, loop=add_self_edges) # get k nearest neighbours. what is batch
        else:
          edge_index = radius_graph(node_features, r=radius, batch=batch_ids, loop=add_self_edges)

        nr_hyperedges = node_features.shape[0] + nr_cl #nr_nodes + nr_clusters
        hyperedge_dim = edge_index.shape[1] + nr_cl * top_s #each node has knn neighbours + each cluster has top S closest nodes. 
        hyperedge_matrix = torch.zeros((2,hyperedge_dim)).to(torch.int64)
        if self.myflags["kmeans_clustering"]:
          clustering = KMeans(n_clusters=nr_cl,n_init=10).fit(node_features.cpu()) # init with k_m_cl cluster. Node features = nr_nodes, 2
          #get KNN nodes to each centre
          for i in range(nr_cl):
              centre = torch.tensor(clustering.cluster_centers_[i]).to(self._device)
              dists = (node_features - centre)
              dists = (dists*dists).sum(dim=1)           # squared dist is equally good as nonsquared dist
              closest_nodes = torch.topk(dists, k=top_s) # top_nodes.indices
              hyperedge_matrix[0,i*top_s:(i+1)*top_s] = closest_nodes.indices
              hyperedge_matrix[1,i*top_s:(i+1)*top_s] = i
        end_pos = nr_cl*top_s
        hyperedge_matrix[0,end_pos:] = edge_index[0,:]#nodes
        hyperedge_matrix[1,end_pos:] = edge_index[1,:]+nr_cl#edges
        return hyperedge_matrix.to(self._device), nr_hyperedges


  def _encoder_preprocessor(
          self,
          position_sequence: torch.tensor,
          nparticles_per_example: torch.tensor,
          particle_types: torch.tensor):
    """Extracts important features from the position sequence. Returns a tuple
    of node_features (nparticles, 30), edge_index (nparticles, nparticles), and
    edge_features (nparticles, 3).

    Args:
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, 6, dim). Includes current + last 5 positions
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles).
    """

    nparticles = position_sequence.shape[0]
    most_recent_position = position_sequence[:, -1]  # (n_nodes, 2)
    velocity_sequence = time_diff(position_sequence)

    # Get connectivity of the graph with shape of (nparticles, 2)
    senders, receivers = self._compute_graph_connectivity(
        most_recent_position, nparticles_per_example, self._connectivity_radius)
    node_features = []

    # Normalized velocity sequence, merging spatial an time axis.
    velocity_stats = self._normalization_stats["velocity"]
    normalized_velocity_sequence = (
        velocity_sequence - velocity_stats['mean']) / velocity_stats['std']
    flat_velocity_sequence = normalized_velocity_sequence.view(
        nparticles, -1)
    # There are 5 previous steps, with dim 2
    # node_features shape (nparticles, 5 * 2 = 10)
    node_features.append(flat_velocity_sequence)

    # Normalized clipped distances to lower and upper boundaries.
    # boundaries are an array of shape [num_dimensions, 2], where the second
    # axis, provides the lower/upper boundaries.
    boundaries = torch.tensor(
        self._boundaries, requires_grad=False).float().to(self._device)
    distance_to_lower_boundary = (
        most_recent_position - boundaries[:, 0][None])
    distance_to_upper_boundary = (
        boundaries[:, 1][None] - most_recent_position)
    distance_to_boundaries = torch.cat(
        [distance_to_lower_boundary, distance_to_upper_boundary], dim=1)
    normalized_clipped_distance_to_boundaries = torch.clamp(
        distance_to_boundaries / self._connectivity_radius, -1., 1.)
    # The distance to 4 boundaries (top/bottom/left/right)
    # node_features shape (nparticles, 10+4)
    node_features.append(normalized_clipped_distance_to_boundaries)

    # Particle type
    if self._nparticle_types > 1:
      particle_type_embeddings = self._particle_type_embedding(
          particle_types)
      node_features.append(particle_type_embeddings)
    # Final node_features shape (nparticles, 30) for 2D
    # 30 = 10 (5 velocity sequences*dim) + 4 boundaries + 16 particle embedding

    # Collect edge features.
    edge_features = []

    # Relative displacement and distances normalized to radius
    # with shape (nedges, 2)
    # normalized_relative_displacements = (
    #     torch.gather(most_recent_position, 0, senders) -
    #     torch.gather(most_recent_position, 0, receivers)
    # ) / self._connectivity_radius
    normalized_relative_displacements = (
        most_recent_position[senders, :] -
        most_recent_position[receivers, :]
    ) / self._connectivity_radius

    # Add relative displacement between two particles as an edge feature
    # with shape (nparticles, ndim)
    edge_features.append(normalized_relative_displacements)

    # Add relative distance between 2 particles with shape (nparticles, 1)
    # Edge features has a final shape of (nparticles, ndim + 1)
    normalized_relative_distances = torch.norm(
        normalized_relative_displacements, dim=-1, keepdim=True)
    edge_features.append(normalized_relative_distances)
 
    if self.myflags["hyper_edge_set"] or self.myflags["USE_BOTH"]:
      #loopless implementation of returning hyperedge set
      nr_edges = senders.shape[0]
      top = torch.zeros((nr_edges*2),dtype=torch.int64,device=self._device)#.to(torch.int64).to(self._device)  #((1,2),(e2),(e3),...) - top is pairs of node indexes indicating edges
      bot = torch.tensor(np.floor(np.arange(0, nr_edges, 0.5))).to(torch.int64).to(self._device)               #(0,0,1,1,2,2,...)     - hyper-edge indexes.
      idx_send = np.arange(0,2*nr_edges,2)
      idx_rec  = np.arange(1,2*nr_edges,2)
      top[idx_send]=senders
      top[idx_rec]=receivers
      top = top.reshape(-1,1)
      bot = bot.reshape(-1,1)
      hyper_edge_indices = torch.transpose(torch.cat((top, bot), dim=-1), 0, 1)
      
      # NEW For every hyperedge, add the node features. This way, we can tell apart edges.
      edge_features = torch.cat(edge_features, dim=-1)
      node_features = torch.cat(node_features, dim=-1)

      
      if False:
        all_edge_ftrs_send = node_features[senders].squeeze()
        all_edge_ftrs_rec = node_features[receivers].squeeze()
        buffa = torch.cat((all_edge_ftrs_send, all_edge_ftrs_rec), dim=1) # edges x 60
        edge_features = torch.cat((buffa,edge_features), dim=1)
      # Calculate hyperedge list for arbitrary hyperedge size
      '''hyper_edge_list, new_hyperedge = [], []
      max_idx = 0

      for i in range(hyper_edge_indices.shape[1]):
        cur_value, cur_idx = hyper_edge_indices[:, i]

        if cur_idx <= max_idx:
          new_hyperedge.append(cur_value)
        else:
          hyper_edge_list.append(new_hyperedge)
          new_hyperedge = [cur_value]
          max_idx += 1

      hyper_edge_list.append(new_hyperedge)'''
      
      # Calculate hyperedge list for 2-uniform hyperedges
      #hyper_edge_list = np.split(hyper_edge_indices[0], nr_edges)
      return (node_features,
              hyper_edge_indices,
              edge_features)

    return (torch.cat(node_features, dim=-1),
            torch.stack([senders, receivers]),
            torch.cat(edge_features, dim=-1))


  def _encoder_preprocessor_hypergraph(#_build_graph_from_raw
          self,
          position_sequence: torch.tensor,
          nparticles_per_example: torch.tensor,
          particle_types: torch.tensor):
      # Build hypergraph as done in https://www.ijcai.org/proceedings/2019/0366.pdf The method uses both KNN and K means clustering
      # Algorithm 1 (Hypergraph construction) This is different from using a connectivity radius, we can incorporate this in future
      # returns node features.                                Shape: (n x f)
      # Indices List ((3,7,9),(2,1),(e3),(e4),...).           Shape: (e)     #where (3,7,9) are node idxs. 
      # hyper edge features                                   Shape: (e x f) #e - nr hyperedges. f - nr of hyperfeatures per edge.
      # Currently 29 hyperedge features - pos: mean (2), std(2), max(2), min(2), "area(1)", vel: mean (5*2), std (5*2)
      n_total_points = position_sequence.shape[0]
      #if True:
      #  return torch.zeros((n_total_points,30)).to(torch.float32).to(self._device), [torch.zeros((6)).to(torch.int32) for i in range(n_total_points)], torch.zeros((n_total_points,29)).to(torch.float32).to(self._device)
      most_recent_position = position_sequence[:, -1] # (n_nodes, 2)
      velocity_sequence = time_diff(position_sequence)
      # senders and receivers are integers of shape (E,)
      hyper_edge_set, nr_edges = self._compute_graph_connectivity_hypergraph(most_recent_position, nparticles_per_example, self._connectivity_radius) # retruns k x 2 edge matrix and nr edges.
      node_features = []
      # Normalized velocity sequence, merging spatial an time axis.
      velocity_stats = self._normalization_stats["velocity"]
      normalized_velocity_sequence = (velocity_sequence - velocity_stats['mean']) / velocity_stats['std']
      flat_velocity_sequence = normalized_velocity_sequence.view(n_total_points, -1)
      node_features.append(flat_velocity_sequence)

      # Normalized clipped distances to lower and upper boundaries.
      # boundaries are an array of shape [num_dimensions, 2], where the second
      # axis, provides the lower/upper boundaries.
      boundaries = torch.tensor(self._boundaries, requires_grad=False).float().to(self._device)
      distance_to_lower_boundary = (most_recent_position - boundaries[:, 0][None])
      distance_to_upper_boundary = (boundaries[:, 1][None] - most_recent_position)
      distance_to_boundaries = torch.cat([distance_to_lower_boundary, distance_to_upper_boundary], dim=1)
      normalized_clipped_distance_to_boundaries = torch.clamp(distance_to_boundaries / self._connectivity_radius, -1., 1.)
      node_features.append(normalized_clipped_distance_to_boundaries)

      if self._nparticle_types > 1:
          particle_type_embeddings = self._particle_type_embedding(particle_types)
          node_features.append(particle_type_embeddings)
      # HYPEREDGE FEATURES
      #Position features (mean, std)
      e_ftrs_pos = torch.index_select(most_recent_position, 0, hyper_edge_set[0,:])
      e_ftrs_pos_mean = scatter_mean(e_ftrs_pos, hyper_edge_set[1,:], dim=0)#ex2 (x,y) (Midpoints)
      e_ftrs_pos_std = scatter_std(e_ftrs_pos, hyper_edge_set[1,:], dim=0)
      #Box features (minx, miny, maxx, maxy)
      e_ftrs_pos_max = scatter_max(e_ftrs_pos, hyper_edge_set[1,:], dim=0)[0]#[0] = get the values, [1] = get indices of nodes which give maximum
      e_ftrs_pos_min = scatter_min(e_ftrs_pos, hyper_edge_set[1,:], dim=0)[0]
      #Area 
      e_ftrs_area = (e_ftrs_pos_max-e_ftrs_pos_min)#xy diff
      e_ftrs_area = (e_ftrs_area[:,0] * e_ftrs_area[:,1]).reshape(-1,1)
      #Velocity features (mean, std) last 5 time-steps
      e_ftrs_vel = torch.index_select(flat_velocity_sequence, 0, hyper_edge_set[0,:])#
      e_ftrs_vel_mean = scatter_mean(e_ftrs_vel, hyper_edge_set[1,:], dim=0)#ex10 5 timesteps * 2 coordinates
      e_ftrs_vel_std = scatter_std(e_ftrs_vel, hyper_edge_set[1,:], dim=0)
      #concat them all :)
      edge_features = torch.cat((e_ftrs_pos_mean, e_ftrs_pos_std, e_ftrs_pos_max, e_ftrs_pos_min, e_ftrs_area, e_ftrs_vel_mean, e_ftrs_vel_std),dim=-1)
      # Leaving the alternative implementation of indices_list
      # indices_list=[]
      # for idx in range(nr_edges):
      #    indices_list.append( hyper_edge_set[0,hyper_edge_set[1,:]==idx] ) # (hyper_edge_set[0,:]==idx).nonzero()#indices_list = [torch.ones((6)).to(torch.int32)*max(i-12,0) for i in range(n_total_points+6)]
      return torch.cat(node_features, dim=-1), hyper_edge_set.to(self._device), edge_features#hyper_edge_set



  def _decoder_postprocessor(
          self,
          normalized_acceleration: torch.tensor,
          position_sequence: torch.tensor) -> torch.tensor:
    """ Compute new position based on acceleration and current position.
    The model produces the output in normalized space so we apply inverse
    normalization.

    Args:
      normalized_acceleration: Normalized acceleration (nparticles, dim).
      position_sequence: Position sequence of shape (nparticles, dim).

    Returns:
      torch.tensor: New position of the particles.

    """
    # Extract real acceleration values from normalized values
    acceleration_stats = self._normalization_stats["acceleration"]
    acceleration = (
        normalized_acceleration * acceleration_stats['std']
    ) + acceleration_stats['mean']

    # Use an Euler integrator to go from acceleration to position, assuming
    # a dt=1 corresponding to the size of the finite difference.
    most_recent_position = position_sequence[:, -1]
    most_recent_velocity = most_recent_position - position_sequence[:, -2]

    # TODO: Fix dt
    new_velocity = most_recent_velocity + acceleration  # * dt = 1
    new_position = most_recent_position + new_velocity  # * dt = 1
    return new_position

  def predict_positions(
          self,
          current_positions: torch.tensor,
          nparticles_per_example: torch.tensor,
          particle_types: torch.tensor) -> torch.tensor:
    """Predict position based on acceleration.

    Args:
      current_positions: Current particle positions (nparticles, dim).
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles).

    Returns:
      next_positions (torch.tensor): Next position of particles.
    """
    

    if self.myflags["USE_BOTH"]:
      #node features are the same, edge_indexes and edge features will differ.
      node_features, edge_index_h, edge_features_h = self._encoder_preprocessor_hypergraph(
        current_positions, nparticles_per_example, particle_types)
      node_features, edge_index_g, edge_features_g = self._encoder_preprocessor(
        current_positions, nparticles_per_example, particle_types)
      
    elif self.myflags["return_hyperedges"]:
      node_features, edge_index, edge_features = self._encoder_preprocessor_hypergraph(
        current_positions, nparticles_per_example, particle_types)
    else:
      node_features, edge_index, edge_features = self._encoder_preprocessor(
        current_positions, nparticles_per_example, particle_types)
    
    if self.myflags["USE_BOTH"]:
      predicted_normalized_acceleration = self._encode_process_decode_both(
        node_features, edge_index_g, edge_features_g, edge_index_h, edge_features_h)
    else:
      predicted_normalized_acceleration = self._encode_process_decode(
        node_features, edge_index, edge_features)
    
    next_positions = self._decoder_postprocessor(
        predicted_normalized_acceleration, current_positions)
    return next_positions

  def predict_accelerations(
          self,
          next_positions: torch.tensor,
          position_sequence_noise: torch.tensor,
          position_sequence: torch.tensor,
          nparticles_per_example: torch.tensor,
          particle_types: torch.tensor):
    """Produces normalized and predicted acceleration targets.

    Args:
      next_positions: Tensor of shape (nparticles_in_batch, dim) with the
        positions the model should output given the inputs.
      position_sequence_noise: Tensor of the same shape as `position_sequence`
        with the noise to apply to each particle.
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, 6, dim). Includes current + last 5 positions.
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles).

    Returns:
      Tensors of shape (nparticles_in_batch, dim) with the predicted and target
        normalized accelerations.

    """

    # Add noise to the input position sequence.
    noisy_position_sequence = position_sequence + position_sequence_noise

    # Perform the forward pass with the noisy position sequence.
    if self.myflags["USE_BOTH"]:
      #node features are the same, edge_indexes and edge features will differ.
      node_features, edge_index_h, edge_features_h = self._encoder_preprocessor_hypergraph(
        noisy_position_sequence, nparticles_per_example, particle_types)
      node_features, edge_index_g, edge_features_g = self._encoder_preprocessor(
        noisy_position_sequence, nparticles_per_example, particle_types)
      

    elif self.myflags["return_hyperedges"]:
      node_features, edge_index, edge_features = self._encoder_preprocessor_hypergraph(
        noisy_position_sequence, nparticles_per_example, particle_types)
    else:
      node_features, edge_index, edge_features = self._encoder_preprocessor(
        noisy_position_sequence, nparticles_per_example, particle_types)
    
    if self.myflags["USE_BOTH"]:
      predicted_normalized_acceleration = self._encode_process_decode_both(
        node_features, edge_index_g, edge_features_g, edge_index_h, edge_features_h)
    else:
      predicted_normalized_acceleration = self._encode_process_decode(
        node_features, edge_index, edge_features)

    # Calculate the target acceleration, using an `adjusted_next_position `that
    # is shifted by the noise in the last input position.
    next_position_adjusted = next_positions + position_sequence_noise[:, -1]
    target_normalized_acceleration = self._inverse_decoder_postprocessor(
        next_position_adjusted, noisy_position_sequence)
    # As a result the inverted Euler update in the `_inverse_decoder` produces:
    # * A target acceleration that does not explicitly correct for the noise in
    #   the input positions, as the `next_position_adjusted` is different
    #   from the true `next_position`.
    # * A target acceleration that exactly corrects noise in the input velocity
    #   since the target next velocity calculated by the inverse Euler update
    #   as `next_position_adjusted - noisy_position_sequence[:,-1]`
    #   matches the ground truth next velocity (noise cancels out).

    return predicted_normalized_acceleration, target_normalized_acceleration

  def _inverse_decoder_postprocessor(
          self,
          next_position: torch.tensor,
          position_sequence: torch.tensor):
    """Inverse of `_decoder_postprocessor`.

    Args:
      next_position: Tensor of shape (nparticles_in_batch, dim) with the
        positions the model should output given the inputs.
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, 6, dim). Includes current + last 5 positions.

    Returns:
      normalized_acceleration (torch.tensor): Normalized acceleration.

    """
    previous_position = position_sequence[:, -1]
    previous_velocity = previous_position - position_sequence[:, -2]
    next_velocity = next_position - previous_position
    acceleration = next_velocity - previous_velocity

    acceleration_stats = self._normalization_stats["acceleration"]
    normalized_acceleration = (
        acceleration - acceleration_stats['mean']) / acceleration_stats['std']
    return normalized_acceleration

  def save(
          self,
          path: str = 'model.pt'):
    """Save model state

    Args:
      path: Model path
    """
    torch.save(self.state_dict(), path)

  def load(
          self,
          path: str):
    """Load model state from file

    Args:
      path: Model path
    """
    self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))


def time_diff(
        position_sequence: torch.tensor) -> torch.tensor:
  """Finite difference between two input position sequence

  Args:
    position_sequence: Input position sequence & shape(nparticles, 6 steps, dim)

  Returns:
    torch.tensor: Velocity sequence
  """
  return position_sequence[:, 1:] - position_sequence[:, :-1]
