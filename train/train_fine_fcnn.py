import click
import torch
import torch.nn as nn
import fcnn
import datasets
import plot
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from os import makedirs
from os.path import join


@click.command()
@click.option('--datafile', default='data/train.csv')
@click.option('--batch-size', default=2)
@click.option('--max-epochs', default=20)  # 100000
@click.option('--height1', default=256)
@click.option('--width1', default=256)  # Dims. of downsampled input for computing sample points.
@click.option('--height2', default=1024)  # Dims. of image from which to sample patches.
@click.option('--width2', default=1024)
@click.option('--lr', default=0.00001)
@click.option('--sample-every', default=10)
@click.option('--checkpoint-every', default=10)
@click.option('--num-workers', default=4)
@click.option('--model-name', default='fine')
@click.option('--pretrain-fpath', default='results/coarse/weights.params')
@click.option('--num-samples', default=24)  # Max. number of pos./neg. patches per image.
@click.option('--num_fixed', default=32)
@click.option('--patch_size', default=128)
def train_pcnn(datafile, batch_size, max_epochs, height1, width1, height2, width2, lr, sample_every, checkpoint_every, num_workers, model_name, pretrain_fpath, num_samples, num_fixed, patch_size):
    gpu_id = None
    use_cuda = True

    results_dir = join('results', model_name)
    weights_fpath = join(results_dir, 'weights.params')
    samples_dir = join(results_dir, 'samples')
    cache_dir = join(results_dir, 'cache')
    makedirs(samples_dir, exist_ok=True)
    makedirs(cache_dir, exist_ok=True)

    patchnet = fcnn.UNet()
    patchnet_optimizer = optim.SGD(
        patchnet.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001
    )
    criterion = nn.CrossEntropyLoss(reduction='elementwise_mean')

    patchnet.load_state_dict(torch.load(pretrain_fpath))

    def stack_patches_and_labels(data):
        synth_patches, real_patches, targets, indices = zip(*data)
        return (
            torch.cat(synth_patches), torch.cat(real_patches),
            torch.cat(targets), np.hstack(indices)
        )

    train_list, valid_list = datasets.split_train_val(datafile)
    print('%d training examples' % len(train_list))
    print('%d validation examples' % len(valid_list))

    train = datasets.FineDataset(
        train_list, cache_dir, height1, width1, height2, width2, patch_size,
        num_samples, num_fixed
    )
    valid = datasets.FineDataset(
        valid_list, cache_dir, height1, width1, height2, width2, patch_size,
        num_samples, num_fixed
    )

    train_iter = DataLoader(
        train, shuffle=True, batch_size=batch_size,
        num_workers=num_workers, collate_fn=stack_patches_and_labels
    )
    valid_iter = DataLoader(
        valid, shuffle=False, batch_size=batch_size,
        num_workers=num_workers, collate_fn=stack_patches_and_labels
    )

    if use_cuda:
        patchnet.cuda(gpu_id)

    print('Starting training.')
    try:
        for epoch in range(1, max_epochs):
            print('Epoch %d' % (epoch))
            # Training
            patchnet.train()
            train_losses = []
            for itr, (x_synth, _, y, _) in enumerate(train_iter):
                if use_cuda:
                    y = y.cuda(gpu_id)
                    x_synth = x_synth.cuda(gpu_id)
                z, y_hat = patchnet(x_synth)
                train_loss = criterion(z, y)

                patchnet_optimizer.zero_grad()
                train_loss.backward()
                patchnet_optimizer.step()
                train_losses.append(train_loss.item())

            print(' Train: loss = %.6f' % np.mean(train_losses))

            # Validation
            patchnet.eval()
            valid_losses = []
            # Sample first epoch to ensure plotting is working.
            visualize = (epoch == 1 or (epoch % sample_every == 0))
            for itr, (x_synth, x_real, y, indices) in enumerate(valid_iter):
                if use_cuda:
                    y = y.cuda(gpu_id)
                    x_synth = x_synth.cuda(gpu_id)
                with torch.no_grad():
                    z, y_hat = patchnet(x_synth)
                    valid_loss = criterion(z, y)
                valid_losses.append(valid_loss.item())

                if visualize:
                    if use_cuda:
                        x_real = x_real.cuda(gpu_id)
                    with torch.no_grad():
                        _, y_hat_real = patchnet(x_real)
                    plot.plot_real_and_synth_samples(
                        x_synth, y, y_hat, x_real, y_hat_real,
                        [join(samples_dir, '%d_%d.jpg' % (itr, i))
                         for i in range(x_synth.shape[0])]
                    )
            print(' Valid: loss = %.6f, visualize = %s' % (
                np.mean(valid_losses), visualize))

            if (epoch % checkpoint_every) == 0:
                checkpoint_fpath = '%s.chkpt' % (weights_fpath)
                print('Saving checkpoint to %s' % (checkpoint_fpath))
                torch.save(patchnet.state_dict(), checkpoint_fpath)

    except KeyboardInterrupt:
        print('Stopped training')
    print('Saving learned weights to %s.' % (weights_fpath))
    torch.save(patchnet.state_dict(), weights_fpath)


if __name__ == '__main__':
    train_pcnn()
