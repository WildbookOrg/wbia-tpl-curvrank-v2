import click
import datasets
import regression
import numpy as np
import plot
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
@click.option('--height', default=224)  # height, width = 336,336
@click.option('--width', default=224)
@click.option('--lr', default=0.001)
@click.option('--sample-every', default=10)
@click.option('--checkpoint-every', default=10)
@click.option('--num-workers', default=4)
@click.option('--model-name', default='anchor')
def train_regression(datafile, batch_size, max_epochs, pad, height, width, lr, sample_every, checkpoint_every, num_workers, model_name):
    gpu_id = None
    use_cuda = True

    results_dir = join('results', model_name)
    weights_fpath = join(results_dir, 'weights.params')
    samples_dir = join(results_dir, 'samples')
    makedirs(samples_dir, exist_ok=True)

    reg_nn = regression.VGG16()

    print('Splitting training/validation by name.')
    train_list, valid_list = datasets.split_train_val(datafile)
    print('%d training examples' % len(train_list))
    print('%d validation examples' % len(valid_list))

    def stack_imgs_and_points(data):
        imgs, imgs_torch, pts0, pts1, indices = zip(*data)
        return (
            imgs, torch.stack(imgs_torch),
            torch.stack(pts0), torch.stack(pts1), np.hstack(indices)
        )

    train = datasets.RegressionDataset(
        train_list, height, width, pad, random_warp=True,
    )
    valid = datasets.RegressionDataset(
        valid_list, height, width, pad, random_warp=False,
    )
    train_iter = DataLoader(
        train, shuffle=True, batch_size=batch_size, num_workers=num_workers,
        collate_fn=stack_imgs_and_points
    )
    valid_iter = DataLoader(
        valid, shuffle=False, batch_size=batch_size, num_workers=num_workers,
        collate_fn=stack_imgs_and_points
    )

    if use_cuda:
        reg_nn.cuda(gpu_id)

    reg_nn_optimizer = optim.SGD(
        reg_nn.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001
    )
    crit1 = nn.MSELoss()
    crit2 = nn.MSELoss()

    print('Starting training.')
    try:
        for epoch in range(1, max_epochs):
            print('Epoch %d' % (epoch))
            # Training
            reg_nn.train()
            train_losses = []
            for itr, (imgs, x, y0, y1, indices) in enumerate(train_iter):
                if use_cuda:
                    x = x.cuda(gpu_id)
                    y0, y1 =  y0.cuda(gpu_id), y1.cuda(gpu_id)
                y0_hat, y1_hat = reg_nn(x)
                reg_nn.zero_grad()
                loss_y0 = crit1(y0_hat, y0)
                loss_y1 = crit2(y1_hat, y1)
                train_loss = 0.5 * (loss_y0 + loss_y1)
                train_loss.backward()
                reg_nn_optimizer.step()
                train_losses.append(train_loss.item())
            print(' Train: loss = %.6f' % (np.mean(train_losses)))
            # Validation
            reg_nn.eval()
            valid_losses = []
            # Sample first epoch to ensure plotting is working.
            visualize = (epoch == 1 or (epoch % sample_every == 0))
            for itr, (imgs, x, y0, y1, indices) in enumerate(valid_iter):
                if use_cuda:
                    x = x.cuda(gpu_id)
                    y0, y1 =  y0.cuda(gpu_id), y1.cuda(gpu_id)
                with torch.no_grad():
                    y0_hat, y1_hat = reg_nn(x)
                    loss_y0 = crit1(y0_hat, y0)
                    loss_y1 = crit2(y1_hat, y1)
                    valid_loss = 0.5 * (loss_y0 + loss_y1)
                valid_losses.append(valid_loss.item())
                # Plot the output to the samples_dir directory.
                if visualize:
                    fpaths = [join(samples_dir, '%d.jpg' % idx)
                              for idx in indices]
                    plot.plot_regression_samples(imgs, y0, y1, y0_hat, y1_hat,
                                                 fpaths)
            print(' Valid: loss = %.6f, visualize = %s' % (
                np.mean(valid_losses), visualize))

            if (epoch % checkpoint_every) == 0:
                checkpoint_fpath = '%s.chkpt' % (weights_fpath)
                print('Saving checkpoint to %s' % (checkpoint_fpath))
                torch.save(reg_nn.state_dict(), checkpoint_fpath)

    except KeyboardInterrupt:
        print('Stopped training')
    print('Saving learned weights to %s.' % (weights_fpath))
    torch.save(reg_nn.state_dict(), weights_fpath)


if __name__ == '__main__':
    train_regression()
