import time

import numpy as np
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import fasterrcnn_resnet50_fpn as fasterrcnn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import torchvision.transforms.functional as F

from matplotlib import pyplot as plt
#plt.rcParams["savefig.bbox"] = 'tight'


from env.env import UEGym


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.show()

def showImagesWithBoxes(images : list, outputs : list):
    score_threshold = 0.8
    images_with_boxes = []

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    print([weights.meta["categories"][label] for output in outputs for label in output['labels']])

    for img, output in zip(images, outputs):
        boxes = output['boxes'][output['scores'] > score_threshold]

        single_img_box = draw_bounding_boxes(
            img,
            boxes=boxes,
            width=4
        )
        images_with_boxes.append(single_img_box)
    show(images_with_boxes)


if __name__ == "__main__":
    print("#" * 50)
    print("\t" * 4, "TEST FAST R-CNN")
    print("#" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}.")

    #### Initialize model ####
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model_v2 = fasterrcnn_resnet50_fpn_v2(weights=weights)
    model_v2.eval()
    model_v2.to(device)
    ##########################
    env = UEGym()

    images, states = env.reset()

    images_normalized = np.moveaxis(images, -1, 1) / 255.0
    images_tensor = torch.tensor(images_normalized, dtype=torch.float)
    images_tensor = images_tensor.to(device)

    start = time.time()
    outputs = model_v2(images_tensor)
    print(f"Time required for {len(images)} images: {time.time() - start}.")


    showImagesWithBoxes(torch.tensor(np.moveaxis(images, -1, 1)), outputs)
