import os
import time

from PIL import Image

import torch
import torchvision
import torch.optim as optim
from torchsummary import summary

from utils import CustomDataSet, sample_data, data_sampler
from models import Generator, UNetGenerator
from stylegan_model import Discriminator
from augmentation import augment
from loss import d_logistic_loss, g_nonsaturating_loss

transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(( 0.5, 0.5, 0.5 ), ( 0.5, 0.5, 0.5 ), inplace=True)
])


def requires_grad(model, flag=True, target_layer=None):
    for name, p in model.named_parameters():
        if target_layer is None or target_layer in name:
            p.requires_grad = flag


def train(
    configs,
    target_loader,
    source_loader,
    g_model,
    d_model,
    g_optim,
    d_optim,
    device,
):

    ###################################################
    ## Generator : face (source) -> cartoon (target) ##
    ###################################################

    target_loader = sample_data(target_loader)
    source_loader = sample_data(source_loader)

    iters = configs["epochs"]
    loss_dict = {}

    start_time = time.time()
    for idx in range(iters):
        ##########################
        ## Update Discriminator ##
        ##########################
        requires_grad(g_model, False)
        requires_grad(d_model, True)

        # time_log = time.time()
        target_img = next(target_loader)
        # print("load image : {}".format(time.time() - time_log))
        # time_log = time.time()

        target_img = target_img.to(device)
        # print("to device : {}".format(time.time() - time_log))
        # time_log = time.time()
        ## Freeze Generator
        ## Do something with generator
        source_img = next(source_loader)
        # print("source load image : {}".format(time.time() - time_log))
        # time_log = time.time()

        source_img = source_img.to(device)
        # print("to device : {}".format(time.time() - time_log))
        # time_log = time.time()

        fake_img = g_model(source_img)    ## Generated Image
        # print("generated shape")
        # print(fake_img.shape)
        # print("generator time : {}".format(time.time() - time_log))
        # time_log = time.time()

        ## ADA (Adaptive Augmentation)
        target_img_aug, _ = augment(target_img, 0.8)
        fake_img_aug, _ = augment(fake_img, 0.8)
        # print("aug time : {}".format(time.time() - time_log))
        # time_log = time.time()

        ## Discriminator Output
        fake_pred = d_model(fake_img_aug)
        real_pred = d_model(target_img_aug)
        # print("dis time : {}".format(time.time() - time_log))
        # time_log = time.time()

        d_loss = d_logistic_loss(real_pred=real_pred, fake_pred=fake_pred)
        # print("loss calculate : {}".format(time.time() - time_log))
        # time_log = time.time()

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        d_model.zero_grad()
        d_loss.backward()
        # print("loss backward : {}".format(time.time() - time_log))
        # time_log = time.time()
        d_optim.step()
        # print("optimizer backward : {}".format(time.time() - time_log))
        # time_log = time.time()
        ##########################
        ## Update Generator     ##
        ##########################

        requires_grad(g_model, True)
        requires_grad(d_model, False)

        fake_img = g_model(source_img)    ## Generated Image
        # print("generator : {}".format(time.time() - time_log))
        # time_log = time.time()

        fake_img_aug, _ = augment(fake_img, 0.8)
        # print("aug : {}".format(time.time() - time_log))
        # time_log = time.time()

        fake_pred = d_model(fake_img_aug)
        # print("dis time : {}".format(time.time() - time_log))
        # time_log = time.time()

        g_loss = g_nonsaturating_loss(fake_pred)
        # print("loss calculate : {}".format(time.time() - time_log))
        # time_log = time.time()

        loss_dict["g"] = g_loss

        g_model.zero_grad()
        g_loss.backward()
        # print("loss backward : {}".format(time.time() - time_log))
        # time_log = time.time()

        g_optim.step()
        # print("optim : {}".format(time.time() - time_log))
        # time_log = time.time()

        print("---------------------")
        if (idx + 1) % 5 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(
                f'Epoch: {idx+1} / Dis Loss: {loss_dict["d"]} / Gen Loss:{loss_dict["g"]} / Time : {elapsed_time} (s)'
            )
            print(
                f'Real score : {loss_dict["real_score"]} | Fake score : {loss_dict["fake_score"]}'
            )
            start_time = time.time()

            if not os.path.isdir(configs["image_save_path"]):
                os.mkdir(configs["image_save_path"])

            with torch.no_grad():
                source_img_path = "./image/" + str(idx + 1) + "_src.png"
                torchvision.utils.save_image(
                    source_img[0, :],
                    source_img_path,
                    normalize=True,
                )
                fake_img_path = "./image/" + str(idx + 1) + "_gen.png"
                torchvision.utils.save_image(
                    fake_img[0, :],
                    fake_img_path,
                    normalize=True,
                )


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: {}".format(device))

    configs = {
        "target_path": "./cartoon_resized/",
        "source_path": "./face_resized/",
        "image_save_path": "./image/",
        "batch_size": 64,
        "learning_rate": 0.001,
        "d_reg_every": 16,
        "g_reg_every": 4,
        "epochs": 2000,
    }

    ## 데이터셋 가져오기

    target_dataset = CustomDataSet(
        main_dir=configs["target_path"],
        transform=transform,
    )
    source_dataset = CustomDataSet(
        main_dir=configs["source_path"],
        transform=transform,
    )

    target_loader = torch.utils.data.DataLoader(
        target_dataset,
        batch_size=configs["batch_size"],
        sampler=data_sampler(target_dataset, shuffle=True),
        drop_last=True,
    )
    source_loader = torch.utils.data.DataLoader(
        source_dataset,
        batch_size=configs["batch_size"],
        sampler=data_sampler(source_dataset, shuffle=True),
        drop_last=True,
    )

    print("Target data size : {}".format(len(target_dataset)))
    print("Source data size : {}".format(len(source_dataset)))

    ## 모델 가져오기

    d_model = Discriminator().to(device)
    # g_model = Generator().to(device)
    g_model = UNetGenerator().to(device)

    # summary(d_model, ( 3, 128, 128 ))
    # summary(g_model, ( 3, 128, 128 ))

    # assert False

    g_reg_ratio = configs["g_reg_every"] / (configs["g_reg_every"] + 1)
    d_reg_ratio = configs["d_reg_every"] / (configs["d_reg_every"] + 1)

    g_optim = optim.Adam(
        g_model.parameters(),
        lr=configs["learning_rate"] * g_reg_ratio,
        betas=(0.9**g_reg_ratio, 0.99**g_reg_ratio),
    )

    d_optim = optim.Adam(
        d_model.parameters(),
        lr=configs["learning_rate"] * d_reg_ratio,
        betas=(0.9**d_reg_ratio, 0.99**d_reg_ratio),
    )

    d_model = d_model.to(device)
    g_model = g_model.to(device)

    train(
        configs, target_loader, source_loader, g_model, d_model, g_optim, d_optim, device
    )
