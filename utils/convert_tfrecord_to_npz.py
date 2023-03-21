import os
import sys
import json
import shutil
import argparse
import functools

import numpy as np
import tensorflow as tf 

import reading_utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Convert tfrecord trajectories to npz.')
    parser.add_argument('--input_dir', default = '/data/inputs/', help='Path to input directory.')
    parser.add_argument('--output_dir', default = '/data/outputs/', help = 'Path to output directory.')
    args = parser.parse_args()

    # Copy metadata file from input to output directory
    metadata_input_file = os.path.join(args.input_dir, 'metadata.json')
    metadata_output_file = os.path.join(args.output_dir, 'metadata.json')
    shutil.copy2(src = metadata_input_file, dst = metadata_output_file)

    # Load metadata
    with open(metadata_output_file, 'rt') as fp:
        metadata = json.loads(fp.read())

    # Iterate over all datasets
    for dataset in ['train', 'valid', 'test']:
        # Load tfrecord dataset
        tf_dataset = tf.data.TFRecordDataset([os.path.join(args.input_dir, dataset + '.tfrecord')])
        tf_dataset = tf_dataset.map(functools.partial(reading_utils.parse_serialized_simulation_example, metadata = metadata))

        # Initialize key and dictionary
        key = 'simulation_trajectory_'
        dict_npz = {}

        # Iterate over all trajectories
        for i, trajectory in enumerate(tf_dataset):
            # Extract positions and particle types
            positions = trajectory[1]['position'][1:, :, :]
            particle_types = trajectory[0]['particle_type']

            # Combine positions and particle types and add them to directory
            try:
                trajectory_npz = np.array([positions, particle_types], dtype = object)
            except Exception as e:
                print(e)

            dict_npz[key + str(i)] = trajectory_npz

        # Save npz dataset to disk
        output_file = os.path.join(args.output_dir, dataset + '.npz')
        np.savez(output_file, **dict_npz)