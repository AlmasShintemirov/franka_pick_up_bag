from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import numpy as np
from scipy.ndimage import gaussian_filter1d
import tqdm, os

READ_PATH = '/media/irobotics/Transcend/isaacsim_data/v5_test/'
SAVE_PATH = '/home/irobotics/Xiwei/franka_pick_up_bag/rlds_dataset_builder/tfds_build/data/'
LEN_DATASET = 5
TRAIN_PERCENTAGE = 0.8
LEN_EPISODE = 20

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
    msg_language = []
    msg_eff = []
    msg_camera = []
    msg_camera_wrist = []
    with Reader(read_path) as reader:
        # iterate over messages
        for connection, _, rawdata in reader.messages():
            if connection.topic == '/language_topic':
                msg = deserialize_cdr(rawdata, connection.msgtype)
                msg_language.append(msg)
            elif connection.topic == '/eff_topic':
                msg = deserialize_cdr(rawdata, connection.msgtype)
                msg_eff.append(msg)
            elif connection.topic == '/Camera_rgb':
                msg = deserialize_cdr(rawdata, connection.msgtype)
                msg_camera.append(msg)
            elif connection.topic == '/Camera_wrist_rgb':
                msg = deserialize_cdr(rawdata, connection.msgtype)
                msg_camera_wrist.append(msg)
            else:
                pass

    epi_len_cam = len(msg_camera)
    epi_len_wrist = len(msg_camera_wrist)
    epi_len_eff = len(msg_eff)
    epi_len_lang = len(msg_language)
    # print("length of camera_rgb: ", epi_len_cam)
    # print("length of camera_wrist: ", epi_len_wrist)
    # print("length of eff: ", epi_len_eff)
    # print("length of language: ", epi_len_lang)

    step_length_camera = int(epi_len_cam/LEN_EPISODE)
    step_length = int(epi_len_lang/LEN_EPISODE)
    step_length_eff = int(epi_len_eff/LEN_EPISODE)

    episode = []

    manager = IteratorManager(
        camera_rgb=msg_camera,
        camera_wrist=msg_camera_wrist,
        eff=msg_eff,
        language=msg_language,
    )

    for i in range(LEN_EPISODE):
        for _ in range(step_length_camera):
            rgb_msg = manager.next('camera_rgb')
            rgb_wrist_msg = manager.next('camera_wrist')

        for _ in range(step_length_eff):
            eff_msg = manager.next('eff')

        for _ in range(step_length):
            language_msg = manager.next('language')
        
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
    
    # smooth_x = gaussian_filter1d(actions[:, 0], sigma=2)
    # smooth_y = gaussian_filter1d(actions[:, 1], sigma=2)
    # smooth_z = gaussian_filter1d(actions[:, 2], sigma=2)
    # smoothed_actions = np.stack([smooth_x, smooth_y, smooth_z, actions[:, 3]], axis=-1)

    for i in range(len(episode)-1):
        #  episode[i]['action'] = smoothed_actions[i]
        #  episode[i]['state'] = smoothed_actions[i] 
        episode[i]['action'] = actions[i+1] - actions[i]
    episode[-1]['action'] = np.zeros_like(actions[-1])

    np.save(save_path, episode)

def main():
    N_TRAIN_EPISODES = int(TRAIN_PERCENTAGE * LEN_DATASET)
    N_VAL_EPISODES = LEN_DATASET - N_TRAIN_EPISODES
    print("Generating train examples...")
    os.makedirs(SAVE_PATH + 'train', exist_ok=True)
    for i in tqdm.tqdm(range(N_TRAIN_EPISODES)):
        create_episode(os.path.join(READ_PATH, f'episode_{i}'), os.path.join(SAVE_PATH, f'train/episode_{i}.npy'))
    
    print("Generating val examples...")
    os.makedirs(SAVE_PATH + 'val', exist_ok=True)
    for i in tqdm.tqdm(range(N_TRAIN_EPISODES, N_TRAIN_EPISODES+N_VAL_EPISODES)):
        create_episode(os.path.join(READ_PATH, f'episode_{i}'), os.path.join(SAVE_PATH, f'val/episode_{i}.npy'))

    print('Successfully created example data!')

if __name__ == "__main__":
    main()