# -*- coding: utf-8 -*-
import click
import wbia_curvrank_v2.train.datasets as datasets
import wbia_curvrank_v2.train.plot as plot
import wbia_curvrank_v2.train.fcnn as fcnn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from os import makedirs
from os.path import join
from torch.utils.data import DataLoader


@click.command()
@click.option('--datafile', default='data/train.csv')
@click.option('--batch-size', default=8)
@click.option('--max-epochs', default=20)
@click.option('--pad', default=0.1)
@click.option('--height', default=256)  # height, width = 192, 384
@click.option('--width', default=256)
@click.option('--lr', default=0.01)
@click.option('--sample-every', default=10)
@click.option('--checkpoint-every', default=10)
@click.option('--num-workers', default=4)
@click.option('--model-name', default='coarse')
def train_fcnn_cmd(
    datafile,
    batch_size,
    max_epochs,
    pad,
    height,
    width,
    lr,
    sample_every,
    checkpoint_every,
    num_workers,
    model_name,
):
    train_fcnn(datafile, batch_size, max_epochs, pad, height, width, lr,
        sample_every, checkpoint_every, num_workers, model_name)


def train_fcnn(
    datafile='data/train.csv',
    batch_size=8,
    max_epochs=20,
    pad=0.1,
    height=256,
    width=256,
    lr=0.01,
    sample_every=10,
    checkpoint_every=10,
    num_workers=4,
    model_name='coarse',
):

    gpu_id = None
    use_cuda = True

    results_dir = join('results', model_name)
    weights_fpath = join(results_dir, 'weights.params')
    samples_dir = join(results_dir, 'samples')
    makedirs(samples_dir, exist_ok=True)

    unet = fcnn.UNet()

    print('Splitting training/validation by name.')
    train_list, valid_list = datasets.split_train_val(datafile)
    print('%d training examples' % len(train_list))
    print('%d validation examples' % len(valid_list))

    train = datasets.CoarseDataset(
        train_list,
        height,
        width,
        pad,
        random_warp=True,
    )
    valid = datasets.CoarseDataset(
        valid_list,
        height,
        width,
        pad,
        random_warp=False,
    )
    train_iter = DataLoader(
        train, shuffle=True, batch_size=batch_size, num_workers=num_workers
    )
    valid_iter = DataLoader(
        valid, shuffle=False, batch_size=batch_size, num_workers=num_workers
    )

    if use_cuda:
        unet.cuda(gpu_id)

    unet_optimizer = optim.SGD(
        unet.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001
    )
    criterion = nn.CrossEntropyLoss(reduction='mean')

    print('Starting training.')
    print('Learning Rate: %.6f' % lr)
    try:
        for epoch in range(1, max_epochs):
            print('Epoch %d' % (epoch))
            # Training
            unet.train()
            train_losses = []
            for itr, (x, y, indices) in enumerate(train_iter):
                if use_cuda:
                    x, y = x.cuda(gpu_id), y.cuda(gpu_id)
                z, y_hat = unet(x)
                unet.zero_grad()
                train_loss = criterion(z, y)
                train_loss.backward()
                unet_optimizer.step()
                train_losses.append(train_loss.item())
            print(' Train: loss = %.6f' % (np.mean(train_losses)))
            # Validation
            unet.eval()
            valid_losses = []
            # Sample first epoch to ensure plotting is working.
            visualize = epoch == 1 or (epoch % sample_every == 0)
            for itr, (x, y, indices) in enumerate(valid_iter):
                if use_cuda:
                    x, y = x.cuda(gpu_id), y.cuda(gpu_id)
                with torch.no_grad():
                    z, y_hat = unet(x)
                    valid_loss = criterion(z, y)
                valid_losses.append(valid_loss.item())
                # Plot the output to the samples_dir directory.
                if visualize:
                    fpaths = [join(samples_dir, '%d.jpg' % idx) for idx in indices]
                    plot.plot_coarse_samples(x, y, y_hat, fpaths)
            print(
                ' Valid: loss = %.6f, visualize = %s' % (np.mean(valid_losses), visualize)
            )

            if (epoch % checkpoint_every) == 0:
                checkpoint_fpath = '%s.chkpt' % (weights_fpath)
                print('Saving checkpoint to %s' % (checkpoint_fpath))
                torch.save(unet.state_dict(), checkpoint_fpath)

    except KeyboardInterrupt:
        print('Stopped training')
    print('Saving learned weights to %s.' % (weights_fpath))
    torch.save(unet.state_dict(), weights_fpath)


if __name__ == '__main__':
    train_fcnn_cmd()
