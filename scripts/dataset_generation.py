import time

import numpy as np

from env import env
from utils import save_image
from definitions import ROOT_DIR

NUMBER_OF_EPISODES = 400
FolderPath = ROOT_DIR + "/dataset/"

if __name__ == "__main__":
    start_time = time.time()

    ue_env = env.UEGym()

    img_cnt = 0
    for i in range(NUMBER_OF_EPISODES):
        images, states = ue_env.reset()
        n_robots = len(ue_env._robot_list)
        n_actions = ue_env.action_size


        for img in images:
            save_image(FolderPath, "img_{0:03d}.png".format(img_cnt), image=img)
            img_cnt += 1

        done = False
        while not done:
            sampled_actions = np.random.uniform(-1, 1, [n_robots, n_actions])
            desired_states = states[:, :n_actions] + sampled_actions

            images, states, rewards, dones = ue_env.step(desired_states)

            for img in images:
                save_image(FolderPath, "img_{0:03d}.png".format(img_cnt), image=img)
                img_cnt += 1

            done = dones[0]


    print(f"Required time: {time.time() - start_time}. Number of images collected: {img_cnt + 1}.")
