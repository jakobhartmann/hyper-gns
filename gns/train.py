import collections
import json
import os
import pickle
import glob
import re
import sys

import wandb
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from absl import flags
from absl import app
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gns import learned_simulator
from gns import noise_utils
from gns import reading_utils
from gns import data_loader
from gns import distribute
#from gns import settings
from utils.custom_logger import Logger

flags.DEFINE_enum(
    'mode', 'train', ['train', 'valid', 'rollout', 'rollout_multiple'],
    help='Train model, validation or rollout evaluation.')
flags.DEFINE_integer('batch_size', 2, help='The batch size.')
flags.DEFINE_float('noise_std', 6.7e-4, help='The std deviation of the noise.')
flags.DEFINE_string('data_path', None, help='The dataset directory.')
flags.DEFINE_string('model_path', 'models/', help=('The path for saving checkpoints of the model.'))
flags.DEFINE_string('output_path', 'rollouts/', help='The path for saving outputs (e.g. rollouts).')
flags.DEFINE_string('model_file', None, help=('Model filename (.pt) to resume from. Can also use "latest" to default to newest file.'))
flags.DEFINE_string('train_state_file', 'train_state.pt', help=('Train state filename (.pt) to resume from. Can also use "latest" to default to newest file.'))

flags.DEFINE_integer('ntraining_steps', int(2E7), help='Number of training steps.')
flags.DEFINE_integer('nsave_steps', int(5000), help='Number of steps at which to save the model.')
flags.DEFINE_integer('train_log_frequency', int(1000), help='Number of training steps after which to log the training progress.')

# Learning rate parameters
flags.DEFINE_float('lr_init', 1e-4, help='Initial learning rate.')
flags.DEFINE_float('lr_decay', 0.1, help='Learning rate decay.')
flags.DEFINE_integer('lr_decay_steps', int(5e6), help='Learning rate decay steps.')

flags.DEFINE_integer("cuda_device_number", None, help="CUDA device (zero indexed), default is None so default CUDA device will be used.")

# Logger
flags.DEFINE_boolean('use_wandb', default = True, help = 'Whether or not to use Weights & Biases for experiment tracking.')
flags.DEFINE_string('wandb_project', default = 'hypergraph-physics', help = 'Weights & Biases project name.')
flags.DEFINE_string('wandb_entity', default = 'camb-mphil', help = 'Weights & Biases entity.')
flags.DEFINE_string('wandb_resume', default = None, help = 'Whether or not to resume a Weights & Biases run.')
flags.DEFINE_string('wandb_run_id', default = None, help = 'Weights & Biases run id to resume a run.')

# Rollout
flags.DEFINE_multi_integer('model_step_list', default = [0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000], help = 'List of steps corresponding to model checkpoints used to perform rollout prediction.')

# Hyperedge type
flags.DEFINE_boolean('USE_BOTH', default = True, help = 'Whether or not to use 2 uniform hyperedges (concat) + hyperedges (kmeans and knn).')
flags.DEFINE_boolean('return_hyperedges', default = True, help = 'Whether to use hyperedges (knn/connectivity radus + kmeans)')
flags.DEFINE_boolean('hyper_edge_set', default = True, help = 'Whether to use 2 uniform hyperedges (NO CONCAT)')

# Which hyperedge features to use
flags.DEFINE_boolean('knn_clustering', default = True, help = 'See settings.py for more info')
flags.DEFINE_boolean('radius_clustering', default = False, help = 'See settings.py for more info')
flags.DEFINE_boolean('kmeans_clustering', default = True, help = 'See settings.py for more info')


# Hyperedge feature settings
flags.DEFINE_integer('k_m_cl', 6, help='Nr clusters.')
flags.DEFINE_integer('top_s', 10, help='Nr top s nodes to choose from each cluster.')
flags.DEFINE_integer('k_nn_nr', 4, help='Nr k nearest neighbours.')
flags.DEFINE_float('con_rad', 0.015, help='Connectivity radius.')





Stats = collections.namedtuple('Stats', ['mean', 'std'])

INPUT_SEQUENCE_LENGTH = 6  # So we can calculate the last 5 velocities.
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3

def rollout(
        simulator: learned_simulator.LearnedSimulator,
        position: torch.tensor,
        particle_types: torch.tensor,
        n_particles_per_example: torch.tensor,
        nsteps: int,
        device):
  """Rolls out a trajectory by applying the model in sequence.

  Args:
    simulator: Learned simulator.
    features: Torch tensor features.
    nsteps: Number of steps.
  """
  initial_positions = position[:, :INPUT_SEQUENCE_LENGTH]
  ground_truth_positions = position[:, INPUT_SEQUENCE_LENGTH:]

  current_positions = initial_positions
  predictions = []

  for step in range(nsteps):
    # Get next position with shape (nnodes, dim)
    next_position = simulator.predict_positions(
        current_positions,
        nparticles_per_example=[n_particles_per_example],
        particle_types=particle_types,
    )

    # Update kinematic particles from prescribed trajectory.
    kinematic_mask = (particle_types == KINEMATIC_PARTICLE_ID).clone().detach().to(device)
    next_position_ground_truth = ground_truth_positions[:, step]
    kinematic_mask = kinematic_mask.bool()[:, None].expand(-1, current_positions.shape[-1])
    next_position = torch.where(
        kinematic_mask, next_position_ground_truth, next_position)
    predictions.append(next_position)

    # Shift `current_positions`, removing the oldest position in the sequence
    # and appending the next position at the end.
    current_positions = torch.cat(
        [current_positions[:, 1:], next_position[:, None, :]], dim=1)

  # Predictions with shape (time, nnodes, dim)
  predictions = torch.stack(predictions)
  ground_truth_positions = ground_truth_positions.permute(1, 0, 2)

  loss = (predictions - ground_truth_positions) ** 2

  output_dict = {
      'initial_positions': initial_positions.permute(1, 0, 2).cpu().numpy(),
      'predicted_rollout': predictions.cpu().numpy(),
      'ground_truth_rollout': ground_truth_positions.cpu().numpy(),
      'particle_types': particle_types.cpu().numpy(),
  }

  return output_dict, loss


def predict(device: str, FLAGS, myflags):
  """Predict rollouts.

  Args:
    simulator: Trained simulator if not will undergo training.

  """
  metadata = reading_utils.read_metadata(FLAGS.data_path)
  simulator = _get_simulator(metadata, FLAGS.noise_std, FLAGS.noise_std, device, myflags)

  # Load simulator
  if os.path.exists(FLAGS.model_path + FLAGS.model_file):
    simulator.load(FLAGS.model_path + FLAGS.model_file)
  else:
    train(simulator)
  
  simulator.to(device)
  simulator.eval()

  # Output path
  if not os.path.exists(FLAGS.output_path):
    os.makedirs(FLAGS.output_path)

  # Use `valid`` set for eval mode if not use `test`
  split = 'test' if FLAGS.mode == 'rollout' else 'valid'

  ds = data_loader.get_data_loader_by_trajectories(path=f"{FLAGS.data_path}{split}.npz")

  eval_loss = []
  with torch.no_grad():
    for example_i, (positions, particle_type, n_particles_per_example) in enumerate(ds):
      positions.to(device)
      particle_type.to(device)
      n_particles_per_example = torch.tensor([int(n_particles_per_example)], dtype=torch.int32).to(device)

      nsteps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH
      # Predict example rollout
      example_rollout, loss = rollout(simulator, positions.to(device), particle_type.to(device),
                                      n_particles_per_example.to(device), nsteps, device)

      example_rollout['metadata'] = metadata
      print("Predicting example {} loss: {}".format(example_i, loss.mean()))
      eval_loss.append(torch.flatten(loss))
      
      # Save rollout in testing
      if FLAGS.mode == 'rollout':
        example_rollout['metadata'] = metadata
        filename = f'rollout_{example_i}.pkl'
        filename = os.path.join(FLAGS.output_path, filename)
        with open(filename, 'wb') as f:
          pickle.dump(example_rollout, f)

  print("Mean loss on rollout prediction: {}".format(
      torch.mean(torch.cat(eval_loss))))
  

def predict_multiple(device: str, FLAGS, myflags):
  """Predict rollouts for multiple models.

  Args:
    simulator: Trained simulator if not will undergo training.

  """
  metadata = reading_utils.read_metadata(FLAGS.data_path)
  simulator = _get_simulator(metadata, FLAGS.noise_std, FLAGS.noise_std, device, myflags)

  logger = Logger(use_wandb = FLAGS.use_wandb, wandb_project = FLAGS.wandb_project, wandb_entity = FLAGS.wandb_entity, wandb_resume = FLAGS.wandb_resume, wandb_run_id = FLAGS.wandb_run_id, config = FLAGS)

  for step in FLAGS.model_step_list:
    model_file = os.path.join(FLAGS.model_path, 'model-' + str(step) + '.pt')
    output_path = os.path.join(FLAGS.output_path, str(step))

    # Load simulator
    if os.path.exists(model_file):
      simulator.load(model_file)
    else:
      train(simulator)
    
    simulator.to(device)
    simulator.eval()

    # Output path
    if not os.path.exists(output_path):
      os.makedirs(output_path)

    # Use `valid`` set for eval mode if not use `test`
    split = 'test' if FLAGS.mode == 'rollout_multiple' else 'valid'

    ds = data_loader.get_data_loader_by_trajectories(path=f"{FLAGS.data_path}{split}.npz")

    eval_loss = []
    with torch.no_grad():
      for example_i, (positions, particle_type, n_particles_per_example) in enumerate(ds):
        positions.to(device)
        particle_type.to(device)
        n_particles_per_example = torch.tensor([int(n_particles_per_example)], dtype=torch.int32).to(device)

        nsteps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH
        # Predict example rollout
        example_rollout, loss = rollout(simulator, positions.to(device), particle_type.to(device),
                                        n_particles_per_example.to(device), nsteps, device)

        example_rollout['metadata'] = metadata
        print("Predicting example {} loss: {}".format(example_i, loss.mean()))
        eval_loss.append(torch.flatten(loss))
        
        # Save rollout in testing
        if FLAGS.mode == 'rollout_multiple':
          example_rollout['metadata'] = metadata
          filename = f'rollout_{example_i}.pkl'
          filename = os.path.join(output_path, filename)
          with open(filename, 'wb') as f:
            pickle.dump(example_rollout, f)

    mean_loss = torch.mean(torch.cat(eval_loss))
    print("Mean loss on rollout prediction: {}".format(mean_loss))
    logger.log(data = {'test_loss': mean_loss}, step = step)
  

def optimizer_to(optim, device):
  for param in optim.state.values():
    # Not sure there are any global tensors in the state dict
    if isinstance(param, torch.Tensor):
      param.data = param.data.to(device)
      if param._grad is not None:
        param._grad.data = param._grad.data.to(device)
    elif isinstance(param, dict):
      for subparam in param.values():
        if isinstance(subparam, torch.Tensor):
          subparam.data = subparam.data.to(device)
          if subparam._grad is not None:
            subparam._grad.data = subparam._grad.data.to(device)

def train(rank, flags, world_size):
  """Train the model.

  Args:
    rank: local rank
    world_size: total number of ranks
  """
  #train(rank, myflags, world_size)
  distribute.setup(rank, world_size)

  metadata = reading_utils.read_metadata(flags["data_path"])
  serial_simulator = _get_simulator(metadata, flags["noise_std"], flags["noise_std"], rank, flags)

  simulator = DDP(serial_simulator.to(rank), device_ids=[rank], output_device=rank)
  optimizer = torch.optim.Adam(simulator.parameters(), lr=flags["lr_init"]*world_size)
  step = 0

  if rank == 0:
    # Initialize logger
    logger = Logger(use_wandb = flags['use_wandb'], wandb_project = flags['wandb_project'], wandb_entity = flags['wandb_entity'], wandb_resume = flags['wandb_resume'], wandb_run_id = flags['wandb_run_id'], config = flags)
    train_loss = torch.tensor([]).to(rank)

  # If model_path does exist and model_file and train_state_file exist continue training.
  if flags["model_file"] is not None:

    if flags["model_file"] == "latest" and flags["train_state_file"] == "latest":
      # find the latest model, assumes model and train_state files are in step.
      fnames = glob.glob(f'{flags["model_path"]}*model*pt')
      max_model_number = 0
      expr = re.compile(".*model-(\d+).pt")
      for fname in fnames:
        model_num = int(expr.search(fname).groups()[0])
        if model_num > max_model_number:
          max_model_number = model_num
      # reset names to point to the latest.
      flags["model_file"] = f"model-{max_model_number}.pt"
      flags["train_state_file"] = f"train_state-{max_model_number}.pt"

    if os.path.exists(flags["model_path"] + flags["model_file"]) and os.path.exists(flags["model_path"] + flags["train_state_file"]):
      # load model
      simulator.module.load(flags["model_path"] + flags["model_file"])

      # load train state
      train_state = torch.load(flags["model_path"] + flags["train_state_file"])
      # set optimizer state
      optimizer = torch.optim.Adam(simulator.module.parameters())
      optimizer.load_state_dict(train_state["optimizer_state"])
      optimizer_to(optimizer, rank)
      # set global train state
      step = train_state["global_train_state"].pop("step") + 1
 
    else:
      msg = f'Specified model_file {flags["model_path"] + flags["model_file"]} and train_state_file {flags["model_path"] + flags["train_state_file"]} not found.'
      raise FileNotFoundError(msg) 

  simulator.train()
  simulator.to(rank)

  dl = distribute.get_data_distributed_dataloader_by_samples(path=f'{flags["data_path"]}train.npz',
                                                             input_length_sequence=INPUT_SEQUENCE_LENGTH,
                                                             batch_size=flags["batch_size"],
                                                            )

  print(f"rank = {rank}, cuda = {torch.cuda.is_available()}")
  not_reached_nsteps = True
  start_time = time.time()
  print("start_time", start_time)
  try:
    while not_reached_nsteps:
      torch.distributed.barrier()
      for ((position, particle_type, n_particles_per_example), labels) in dl:
        position.to(rank)
        particle_type.to(rank)
        n_particles_per_example.to(rank)
        labels.to(rank)

        # TODO (jpv): Move noise addition to data_loader
        # Sample the noise to add to the inputs to the model during training.
        sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(position, noise_std_last_step=flags["noise_std"]).to(rank)
        non_kinematic_mask = (particle_type != KINEMATIC_PARTICLE_ID).clone().detach().to(rank)
        sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

        # Get the predictions and target accelerations.
        pred_acc, target_acc = simulator.module.predict_accelerations(
            next_positions=labels.to(rank),
            position_sequence_noise=sampled_noise.to(rank),
            position_sequence=position.to(rank),
            nparticles_per_example=n_particles_per_example.to(rank),
            particle_types=particle_type.to(rank))

        # Calculate the loss and mask out loss on kinematic particles
        loss = (pred_acc - target_acc) ** 2
        loss = loss.sum(dim=-1)
        num_non_kinematic = non_kinematic_mask.sum()
        loss = torch.where(non_kinematic_mask.bool(),
                         loss, torch.zeros_like(loss))
        loss = loss.sum() / num_non_kinematic

        
        # Computes the gradient of loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update learning rate
        lr_new = flags["lr_init"] * (flags["lr_decay"] ** (step/flags["lr_decay_steps"])) * world_size
        for param in optimizer.param_groups:
          param['lr'] = lr_new

        if rank == 0:
          print(f'Training step: {step}/{flags["ntraining_steps"]}. Loss: {loss}.')
          train_loss = torch.cat((train_loss, torch.tensor([loss]).to(rank)))

          if step % flags["train_log_frequency"] == 0:
            logger.log(data = {'train_loss': torch.mean(train_loss)}, step = step)
            train_loss = torch.tensor([]).to(rank)

        # Save model state
        if step % flags["nsave_steps"] == 0 and rank == 0:
          simulator.module.save(flags["model_path"] + 'model-'+str(step)+'.pt')
          train_state = dict(optimizer_state=optimizer.state_dict(), global_train_state={"step":step})
          torch.save(train_state, f'{flags["model_path"]}train_state-{step}.pt')

        # Complete training
        if (step >= flags["ntraining_steps"]):
          not_reached_nsteps = False
          break

        step += 1

  except KeyboardInterrupt:
    pass

  if rank == 0:
    simulator.module.save(flags["model_path"] + 'model-'+str(step)+'.pt')
    train_state = dict(optimizer_state=optimizer.state_dict(), global_train_state={"step":step})
    torch.save(train_state, f'{flags["model_path"]}train_state-{step}.pt')

  distribute.cleanup()
  print("elapsed time: ", start_time - time.time())

def _get_simulator(
        metadata: json,
        acc_noise_std: float,
        vel_noise_std: float,
        device: str,
        myflags) -> learned_simulator.LearnedSimulator:
  
  """Instantiates the simulator.

  Args:
    metadata: JSON object with metadata.
    acc_noise_std: Acceleration noise std deviation.
    vel_noise_std: Velocity noise std deviation.
    device: PyTorch device 'cpu' or 'cuda'.
  """

  # Normalization stats
  normalization_stats = {
      'acceleration': {
          'mean': torch.FloatTensor(metadata['acc_mean']).to(device),
          'std': torch.sqrt(torch.FloatTensor(metadata['acc_std'])**2 +
                            acc_noise_std**2).to(device),
      },
      'velocity': {
          'mean': torch.FloatTensor(metadata['vel_mean']).to(device),
          'std': torch.sqrt(torch.FloatTensor(metadata['vel_std'])**2 +
                            vel_noise_std**2).to(device),
      },
  }

  print("MY_FLAGS", myflags)
  if myflags["USE_BOTH"]:
    simulator = learned_simulator.LearnedSimulator(
      particle_dimensions=metadata['dim'],
      nnode_in=37 if metadata['dim'] == 3 else 30,
      nedge_in=3,#3 features for regular graph
      nedge_in_h=29,#29 features for hypergraph
      latent_dim=128,
      nmessage_passing_steps=10,
      nmlp_layers=2,
      mlp_hidden_dim=128,
      connectivity_radius=metadata['default_connectivity_radius'],
      boundaries=np.array(metadata['bounds']),
      normalization_stats=normalization_stats,
      nparticle_types=NUM_PARTICLE_TYPES,
      particle_type_embedding_size=16,
      device=device,
      myflags = myflags)
  elif myflags["return_hyperedges"]:#if we are using hyperedge, we have 29 edge features. 
    simulator = learned_simulator.LearnedSimulator(
      particle_dimensions=metadata['dim'],
      nnode_in=37 if metadata['dim'] == 3 else 30,
      nedge_in=29,#metadata['dim'] + 1
      latent_dim=128,
      nmessage_passing_steps=10,
      nmlp_layers=2,
      mlp_hidden_dim=128,
      connectivity_radius=metadata['default_connectivity_radius'],
      boundaries=np.array(metadata['bounds']),
      normalization_stats=normalization_stats,
      nparticle_types=NUM_PARTICLE_TYPES,
      particle_type_embedding_size=16,
      device=device,
      myflags = myflags)
  elif myflags["hyper_edge_set"]:#if we are using 2-uniform hyperedges. Added edge
      simulator = learned_simulator.LearnedSimulator(
      particle_dimensions=metadata['dim'],
      nnode_in=37 if metadata['dim'] == 3 else 30, 
      nedge_in=3,#3 edge ftrs + 2*30 node features.#+2*30
      latent_dim=128,
      nmessage_passing_steps=10,
      nmlp_layers=2,
      mlp_hidden_dim=128,
      connectivity_radius=metadata['default_connectivity_radius'],
      boundaries=np.array(metadata['bounds']),
      normalization_stats=normalization_stats,
      nparticle_types=NUM_PARTICLE_TYPES,
      particle_type_embedding_size=16,
      device=device,
      myflags=myflags)
  else:
    simulator = learned_simulator.LearnedSimulator(
        particle_dimensions=metadata['dim'],
        nnode_in=37 if metadata['dim'] == 3 else 30,
        nedge_in=metadata['dim'] + 1,
        latent_dim=128,
        nmessage_passing_steps=10,#10
        nmlp_layers=2,
        mlp_hidden_dim=128,
        connectivity_radius=metadata['default_connectivity_radius'],
        boundaries=np.array(metadata['bounds']),
        normalization_stats=normalization_stats,
        nparticle_types=NUM_PARTICLE_TYPES,
        particle_type_embedding_size=16,
        device=device,
        myflags=myflags)

  return simulator


def main(_):
  """Train or evaluates the model.

  """
  
  
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "29500"
  FLAGS = flags.FLAGS
  myflags = {}
  myflags["data_path"] = FLAGS.data_path
  myflags["noise_std"] = FLAGS.noise_std
  myflags["lr_init"] = FLAGS.lr_init
  myflags["lr_decay"] = FLAGS.lr_decay
  myflags["lr_decay_steps"] = FLAGS.lr_decay_steps
  myflags["batch_size"] = FLAGS.batch_size
  myflags["ntraining_steps"] = FLAGS.ntraining_steps
  myflags["nsave_steps"] = FLAGS.nsave_steps
  myflags["train_log_frequency"] = FLAGS.train_log_frequency
  myflags["model_file"] = FLAGS.model_file
  myflags["model_path"] = FLAGS.model_path
  myflags["train_state_file"] = FLAGS.train_state_file
  myflags["use_wandb"] = FLAGS.use_wandb
  myflags["wandb_project"] = FLAGS.wandb_project
  myflags["wandb_entity"] = FLAGS.wandb_entity
  myflags["wandb_resume"] = FLAGS.wandb_resume
  myflags["wandb_run_id"] = FLAGS.wandb_run_id
  myflags["model_step_list"] = FLAGS.model_step_list
  #hypergns
  myflags["USE_BOTH"] = FLAGS.USE_BOTH
  myflags["return_hyperedges"] = FLAGS.return_hyperedges
  myflags["hyper_edge_set"] = FLAGS.hyper_edge_set
  myflags["knn_clustering"] = FLAGS.knn_clustering
  myflags["radius_clustering"] = FLAGS.radius_clustering
  myflags["kmeans_clustering"] = FLAGS.kmeans_clustering
  myflags["k_m_cl"] = FLAGS.k_m_cl
  myflags["top_s"] = FLAGS.top_s
  myflags["k_nn_nr"] = FLAGS.k_nn_nr
  myflags["con_rad"] = FLAGS.con_rad
  
  
  # Read metadata
  if FLAGS.mode == 'train':
    # If model_path does not exist create new directory.
    if not os.path.exists(FLAGS.model_path):
      os.makedirs(FLAGS.model_path)

    world_size = torch.cuda.device_count()
    print(f"world_size = {world_size}")
    distribute.spawn_train(train, myflags, world_size)

  elif FLAGS.mode in ['valid', 'rollout']:
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if FLAGS.cuda_device_number is not None and torch.cuda.is_available():
      device = torch.device(f'cuda:{int(FLAGS.cuda_device_number)}')
    predict(device, FLAGS, myflags)
  
  elif FLAGS.mode == 'rollout_multiple':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if FLAGS.cuda_device_number is not None and torch.cuda.is_available():
      device = torch.device(f'cuda:{int(FLAGS.cuda_device_number)}')
    predict_multiple(device, FLAGS, myflags)

if __name__ == '__main__':
  app.run(main)
