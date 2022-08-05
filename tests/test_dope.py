import numpy as np
import torch

import dope_models
from env import env

import time

if __name__ == "__main__":

    print("---DOPE test---------------------------------------------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.cuda.empty_cache()
    #model = BetaVAE.BetaVAE(in_channels=3, latent_dim=128)
    #model.to(device)
    ue_env = env.UEGym()

    images, states = ue_env.reset()



    time_list = []
    images_list = []
    for i in range(10):
        start = time.time()

        n_robots = len(ue_env._robot_list)
        initial_pose_array = np.asarray([ue_env._robot_initial_pose * n_robots],
                                        dtype=np.float64).reshape(n_robots, ue_env.action_size)

        observation = ue_env.step(initial_pose_array)
        images_list.append(observation[0])
        time_list.append(time.time() - start)



    time_np = np.array(time_list)
    print(f"------------------"
          f"\nTIME statistics:\nMean: {np.mean(time_np)}\nStd: {np.std(time_np)}"
          f"\n----------------")

    images_normalized = np.moveaxis(images, -1, 1) / 255.0
    images_tensor = torch.tensor(images_normalized, dtype=torch.float32)
    images_tensor = images_tensor.to(device)
    print(len(images))
    print("Stating DOPE")
    start = time.time()
    #outputs = model(images_tensor)
    print(f"Time required for {len(images)} images: {time.time() - start}.")
