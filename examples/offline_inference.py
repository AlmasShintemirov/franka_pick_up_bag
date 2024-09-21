import rclpy
import matplotlib.pyplot as plt
import numpy as np
from subscriber import Subscriber, get_observation
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32MultiArray
from time import sleep

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import cv2
import jax
import tensorflow_datasets as tfds
import tqdm
import mediapy
import numpy as np

VERSION = "v4"
WINDOW_SIZE = 2

PATH_CHECKPOINTS = "/home/irobotics/usb/finetuned_checkpoints/"+VERSION+"_checkpoints/"
PATH_DATASET_TFDS = "/media/irobotics/Transcend/tensorflow_datasets/"+VERSION+"_test/example_dataset/1.0.0/"
PATH_INFERENCE_RESULTS = "/home/irobotics/usb/inference_result/"

### Load the pretrained model
from octo.model.octo_model import OctoModel
model = OctoModel.load_pretrained(PATH_CHECKPOINTS)

### Load the offline dataset
# create RLDS dataset builder
builder = tfds.builder_from_directory(builder_dir=PATH_DATASET_TFDS)
ds = builder.as_dataset(split='train[:2]')
iterator = iter(ds)
episode = next(iterator)
# sample episode + resize to 256x256 (default third-person cam resolution)
steps = list(episode['steps'])
images = [cv2.resize(np.array(step['observation']['image']), (256, 256)) for step in steps]
images_wrist = [np.array(step['observation']['wrist_image']) for step in steps]
# extract goal image & language instruction
goal_image = images[-1]
language_instruction = steps[100]['language_instruction'].numpy().decode()


### Inference
task = model.create_tasks(goals={"image_primary": goal_image})
task = model.create_tasks(texts=[language_instruction])

# run inference loop, this model only uses 3rd person image observations for bridge
# collect predicted and true actions
pred_actions, true_actions = [], []
for step in tqdm.trange(len(images) - (WINDOW_SIZE - 1)):
    input_images = np.stack(images[step:step+WINDOW_SIZE])[None]
    input_images_wrist = np.stack(images_wrist[step:step+WINDOW_SIZE])[None]
    observation = {
        'image_primary': input_images,
        'image_wrist': input_images_wrist,
        'timestep_pad_mask': np.full((1, input_images.shape[1]), True, dtype=bool)
    }
    
    # this returns *normalized* actions --> we need to unnormalize using the dataset statistics
    actions = model.sample_actions(
        observation, 
        task, 
        unnormalization_statistics=model.dataset_statistics["action"], 
        rng=jax.random.PRNGKey(0)
    )
    actions = actions[0] # remove batch dim

    pred_actions.append(actions)
    final_window_step = step + WINDOW_SIZE - 1
    true_actions.append(
        steps[final_window_step]['action']
    )

# save results
np.save(PATH_INFERENCE_RESULTS + VERSION + "_pred_actions.npy", pred_actions[:, 0, :])
np.save(PATH_INFERENCE_RESULTS + VERSION + "_true_actions.npy", true_actions)