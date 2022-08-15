import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as Datasets
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.models as models
import torchvision.utils as vutils
from torch.hub import load_state_dict_from_url

import os
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image


from models.vae.vae import VAE as VAE, vae_loss
from models.vae.vgg19_simp import VGG19 as VGG
from dataset_class import CustomDataset

from definitions import ROOT_DIR


config = {
    "data_dir": ROOT_DIR + "/dataset",
    "batch_size": 16,
    "image_size": 512,
    "lr": 1e-4,
    "epochs": 100,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "save_dir": ROOT_DIR + "/output"
}

if __name__ == "__main__":
    torch.cuda.empty_cache()

    """
    Padding will add to top/bottom
    Image_size servers as new img size we are going to resize
    Then image is randomly cropped to the two times smaller image size
    """
    train_ds = CustomDataset(
        root= config["data_dir"] + "/train",
        image_size=config["image_size"],
        padding=[0, abs(640-480)]

    )

    test_ds = CustomDataset(
        root= config["data_dir"] + "/test",
        image_size=config["image_size"],
        padding=[0, abs(640-480)],

    )


    train_dataloader = DataLoader(dataset=train_ds,
                                  shuffle=True,
                                  batch_size=config["batch_size"],
                                  pin_memory=True,
                                  num_workers=4)

    test_dataloader = DataLoader(dataset=test_ds,
                                 shuffle=True,
                                 batch_size=config["batch_size"],
                                 pin_memory=True,
                                 num_workers=4)

    # Create the feature loss module

    # load the state dict for vgg19
    state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')
    # manually create the feature extractor from vgg19
    feature_extractor = VGG(channel_in=3)

    # loop through the loaded state dict and our vgg19 features net,
    # loop will stop when net.parameters() runs out - so we never get to the "classifier" part of vgg
    for ((name, source_param), target_param) in zip(state_dict.items(), feature_extractor.parameters()):
        target_param.data = source_param.data
        target_param.requires_grad = False

    feature_extractor = feature_extractor.to(config["device"])

    #Create VAE network
    vae_net = VAE(channel_in=3, ch=64, z=1024).to(config["device"])
    # setup optimizer
    optimizer = optim.Adam(vae_net.parameters(), lr=config["lr"], betas=(0.5, 0.999))
    #Loss function
    BCE_Loss = nn.BCEWithLogitsLoss()
    loss_log = []

    #Create the save directory if it does note exist
    if not os.path.isdir(config["save_dir"] + "/Models"):
        os.makedirs(config["save_dir"] + "/Models")
    if not os.path.isdir(config["save_dir"] + "/Results"):
        os.makedirs(config["save_dir"] + "/Results")

    for epoch in range(config["epochs"]):
        vae_net.train()
        print(f"Epoch [{epoch}/{config['epochs']}]:")


        for i, images in enumerate(train_dataloader):
            recon_data, mu, logvar = vae_net(images.to(config["device"]))
            #VAE loss
            loss = vae_loss(recon_data, images.to(config["device"]), mu, logvar)

            # Perception loss
            loss += feature_extractor(torch.cat((torch.sigmoid(recon_data), images.to(config["device"])), 0))

            loss_log.append(loss.item())
            vae_net.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Moving average of loss: {np.average(loss_log[-100:])}")


        vae_net.eval()
        with torch.no_grad():
            for i, image in enumerate(test_ds):
                recon_data, _, _ = vae_net(torch.unsqueeze(image, 0).to(config["device"]))

                if not os.path.isdir( f"{config['save_dir']}/Results/Epoch_{epoch}/"):
                    os.makedirs( f"{config['save_dir']}/Results/Epoch_{epoch}/")

                vutils.save_image(torch.cat((torch.sigmoid(recon_data.cpu()), torch.unsqueeze(image,0)), 2),
                              f"{config['save_dir']}/Results/Epoch_{epoch}/VAE19_img_{i:03d}.png" )

            # Save a checkpoint
            torch.save({
                'epoch': epoch,
                'loss_log': loss_log,
                'model_state_dict': vae_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()

            }, config["save_dir"] + "/Models/" + "VAE19" + "_" + str(config["image_size"]) + ".pt")

