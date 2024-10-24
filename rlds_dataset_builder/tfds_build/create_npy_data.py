import numpy as np
from scipy.ndimage import gaussian_filter1d
import tqdm
import rosbag
import os

READ_PATH = '/scratch/work/zhaox9/isaacsim_data/v4_test/'
SAVE_PATH = '/scratch/work/zhaox9/franka_pick_up_bag/rlds_dataset_builder/tfds_build/data/'
LEN_DATASET = 5
TRAIN_PERCENTAGE = 0.8

class IteratorManager:
    def __init__(self, **iterables):
        self.iterables = iterables
        self.iterators = {key: iter(value) for key, value in iterables.items()}

    def next(self, key):
        try:
            return next(self.iterators[key])
        except StopIteration:
            self.iterators[key] = iter(self.iterables[key])
            return next(self.iterators[key])

    def reset(self, key):
        if key in self.iterables:
            self.iterators[key] = iter(self.iterables[key])
        else:
            raise KeyError(f"No iterable found for key: {key}")

def create_episode(read_path, save_path):

    bag = rosbag.Bag(read_path)
    info = bag.get_type_and_topic_info()
    epi_len_cam = info.topics['/Camera_rgb'].message_count
    epi_len_wrist = info.topics['/Camera_wrist_rgb'].message_count
    epi_len_eff = info.topics['/eff_topic'].message_count
    epi_len_lang = info.topics['/language_topic'].message_count

    epi_len = min(epi_len_cam, epi_len_eff, epi_len_lang, epi_len_wrist)
    step_length = int(epi_len_lang/epi_len)
    step_length_eff = int(epi_len_eff/epi_len)

    # epi_len = min(epi_len_cam, epi_len_joint, epi_len_wrist)

    episode = []
   # tmp_eff = np.zeros(3)

    manager = IteratorManager(
        camera_rgb=bag.read_messages(topics=['/Camera_rgb']),
        camera_wrist=bag.read_messages(topics=['/Camera_wrist_rgb']),
        eff=bag.read_messages(topics=['/eff_topic']),
        language=bag.read_messages(topics=['/language_topic']),
    )

    for i in range(epi_len):
        rgb_msg = manager.next('camera_rgb')[1]
        rgb_wrist_msg = manager.next('camera_wrist')[1]
        for _ in range(step_length_eff):
            eff_msg = manager.next('eff')[1]
       # delta_eff = np.array(eff_msg.data) - np.array(tmp_eff)
       # tmp_eff = eff_msg.data

        for _ in range(step_length):
            language_msg = manager.next('language')[1]
        
        episode.append({
            'image': np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(rgb_msg.height, rgb_msg.width, 3),
            'wrist_image': np.frombuffer(rgb_wrist_msg.data, dtype=np.uint8).reshape(rgb_wrist_msg.height, rgb_wrist_msg.width, 3),
            'state': np.asarray(list(eff_msg.data), dtype=np.float32),
            'action': np.asarray(list(eff_msg.data), dtype=np.float32),
            'language_instruction': str(language_msg.data),
           # 'language_instruction': "pick up the yellow cube and put it to the bag.",
        })
    
    manager.reset('camera_rgb')
    manager.reset('camera_wrist')
    manager.reset('eff')
    manager.reset('language')
    
    actions = np.array([entry['action'] for entry in episode])
    
    smooth_x = gaussian_filter1d(actions[:, 0], sigma=2)
    smooth_y = gaussian_filter1d(actions[:, 1], sigma=2)
    smooth_z = gaussian_filter1d(actions[:, 2], sigma=2)
    smoothed_actions = np.stack([smooth_x, smooth_y, smooth_z, actions[:, 3]], axis=-1)

    for i in range(len(episode)):
         episode[i]['action'] = smoothed_actions[i]
         episode[i]['state'] = smoothed_actions[i] 

    np.save(save_path, episode)
    bag.close()

N_TRAIN_EPISODES = int(TRAIN_PERCENTAGE * LEN_DATASET)
N_VAL_EPISODES = LEN_DATASET - N_TRAIN_EPISODES

# create fake episodes for train and validation
print("Generating train examples...")
os.makedirs(SAVE_PATH + 'train', exist_ok=True)
for i in tqdm.tqdm(range(N_TRAIN_EPISODES)):
    create_episode(read_path=READ_PATH + f'episode_{i}.bag', save_path=SAVE_PATH + f'train/episode_{i}.npy')

print("Generating val examples...")
os.makedirs(SAVE_PATH + 'val', exist_ok=True)
for i in tqdm.tqdm(range(N_TRAIN_EPISODES, N_TRAIN_EPISODES+N_VAL_EPISODES)):
    create_episode(read_path=READ_PATH + f'episode_{i}.bag',save_path=SAVE_PATH + f'val/episode_{i}.npy')

print('Successfully created example data!')

    
