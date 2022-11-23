import os
import time

from PIL import Image

import torch
import torchvision
import torch.optim as optim

from .tools.utils import imshow, imshow_double, CustomDataSet
from .stylegan_model import Discriminator, Generator

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: {}".format(device))

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip((0.3)),
        torchvision.transforms.Resize(( 128, 128 ), interpolation=Image.NEAREST),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(( 0.5, 0.5, 0.5 ), ( 0.5, 0.5, 0.5 ))
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(( 128, 128 ), interpolation=Image.NEAREST),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(( 0.5, 0.5, 0.5 ), ( 0.5, 0.5, 0.5 ))
    ])

    batch_size = 32

    target_dataset = CustomDataSet(main_dir='./cartoon_face/', transform=train_transform)
    source_dataset = CustomDataSet(main_dir='./real_face/', transform=train_transform)

    target_loader = torch.utils.data.DataLoader(
        target_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    source_loader = torch.utils.data.DataLoader(
        source_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    print("Target data size : {}".format(len(target_dataset)))
    print("Source data size : {}".format(len(source_dataset)))

    d_model = Discriminator(size=4)
    g_model = Generator()

    LEARNING_RATE = 0.001
    BETA_1 = 0.9
    BETA_2 = 0.99

    params = list(d_model.parameters()) + list(g_model.parameters())
    optimizer = torch.optim.Adam(
        params=params, lr=LEARNING_RATE, betas=( BETA_1, BETA_2 )
    )

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda epoch: 0.95**epoch,
        last_epoch=-1,
        verbose=False
    )

    d_model = d_model.to(device)
    g_model = g_model.to(device)

    epochs = 200

    ## Per Epoch
    for epoch in range(epochs):

        ## Mini-batch
        for i, _ in enumerate(target_loader):
            optimizer.zero_grad()

            start_time = time.time()

            target_images = next(iter(target_loader))
            source_images = next(iter(source_loader))

            target_images = target_images.to(device)
            source_images = source_images.to(device)

            g_out_source = g_model(source_images)

            d_out_target = d_model(target_images)
            d_out_source = d_model(g_out_source)

            dis_loss_target = discriminator_loss(d_out_target, True)
            dis_loss_source = discriminator_loss(d_out_source, False)
            dis_loss = dis_loss_target + dis_loss_source

            gen_loss = discriminator_loss(d_out_source, True)

            adv_loss = 0.5 * dis_loss + gen_loss

            con_loss = content_loss(source_images, g_out_source)

            weight_param = 0.001
            total_loss = adv_loss + con_loss * weight_param

            total_loss.backward()
            optimizer.step()

            end_time = time.time()
            elapsed_time = end_time - start_time

            if i == len(target_loader) - 1:
                print(
                    'Epoch: %d / Total Loss: %.3f / Adv Loss: %.3f / Time : %.2f (s)' %
                    (epoch + 1, total_loss, adv_loss, elapsed_time)
                )

                if not os.path.isdir("./image/"):
                    os.mkdir("./image/")

                imshow_double(
                    source_images[0, :],
                    g_out_source[0, :],
                    "./image/epoch_" + str(epoch + 1),
                    show=False,
                    save=True
                )

        scheduler.step()