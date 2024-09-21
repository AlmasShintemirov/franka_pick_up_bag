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

PATH_CHECKPOINTS = "/media/irobotics/Transcend/finetuned_checkpoints/v4_checkpoints/"
PATH_DATASET_TFDS = '/media/irobotics/Transcend/tensorflow_datasets/v4_test/example_dataset/1.0.0/'
PATH_INFERENCE_RESULTS = "/media/irobotics/Transcend/inference_result/"


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
# extract goal image & language instruction
goal_image = images[-1]
language_instruction = steps[100]['language_instruction'].numpy().decode()

### Inference
# create `task` dict
# task = model.create_tasks(goals={"image_primary": goal_image[None]})   # for goal-conditioned
task = model.create_tasks(goals={"image_primary": goal_image})
task = model.create_tasks(texts=[language_instruction])                  # for language conditioned

rclpy.init(args=None)
node = Subscriber()
publisher = node.create_publisher(Float32MultiArray, 'online_eff_topic', 10)
pub_msg = Float32MultiArray()

# run inference loop, this model only uses 3rd person image observations for bridge
# collect predicted and true actions
pred_actions= []
# true_actions = steps[:]['action']
while True:

    input_images = get_observation(node, Image, '/Camera_rgb', (256, 256, 3))
    input_images_wrist = get_observation(node, Image, '/Camera_wrist_rgb', (512, 512, 3))

    observation = {
        'image_primary': input_images,
        'image_wrist': input_images_wrist,
        'timestep_pad_mask': np.full((1, input_images.shape[1]), True, dtype=bool)
    }

    # one step actions
    actions = model.sample_actions(
        observation, 
        task, 
        unnormalization_statistics=model.dataset_statistics["action"], 
        rng=jax.random.PRNGKey(0)
    )
    actions = actions[0] # remove batch dim

    # pred_actions.append(actions)
    
    # publish actions to robot
    pub_msg.data = actions[0,:].tolist()

    for i in range(100):
        publisher.publish(pub_msg)
        sleep(0.1)

    # TODO: how to check if episode is done, then break


# node.destroy_node()
# rclpy.shutdown()

