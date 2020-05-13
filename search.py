""" Search cell """
import os
import torch
import utils
import random

import numpy as np
import torch.nn as nn

from config import SearchConfig
from tensorboardX import SummaryWriter
from models.search_cnn import SearchCNNController


config = SearchConfig()
device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)


def main():
    logger.info("Logger is set - training start")

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_size, input_channels, n_classes, train_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=False)

    net_crit = nn.CrossEntropyLoss().to(device)
    model = SearchCNNController(input_size, input_channels, config.init_channels, n_classes,
                                config.layers, net_crit, device_ids=config.gpus)
    model = model.to(device)

    # weights optimizer, weight decay is computed later in `train()`
    w_optim = torch.optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum, weight_decay=0.)
    # alphas optimizer
    alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(0.5, 0.999), weight_decay=0.)
    
    # dataloader, we use the whole training data to search
    n_train = len(train_data)
    indices = list(range(n_train))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=train_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_optim, config.epochs, eta_min=config.w_lr_min)

    # training loop
    for epoch in range(config.epochs):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]

        drop_rate = 0. if epoch < config.warmup_epochs else config.drop_rate
        logger.info("Current drop rate: {:.6f}".format(drop_rate))
        model.print_alphas(logger)

        # training
        train(train_loader, model, w_optim, alpha_optim, lr, epoch, drop_rate)

        # log genotype
        genotype = model.genotype()
        logger.info("genotype = {}".format(genotype))
        with open(os.path.join(config.path, 'genotype.txt'), 'w') as f:
            f.write(str(genotype))

        utils.save_checkpoint(model, config.path, True)
        print()


def train(train_loader, model, w_optim, alpha_optim, lr, epoch, drop_rate):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch * len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)

    model.train()

    for step, (trn_X, trn_y) in enumerate(train_loader):
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        N = trn_X.size(0)
        
        # forward pass loss
        alpha_optim.zero_grad()
        w_optim.zero_grad()

        logits = model(trn_X, drop_rate=drop_rate)
        loss_1 = model.criterion(logits, trn_y)
        loss_1.backward()
        
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)  # gradient clipping
        w_optim.step()
        if epoch >= config.warmup_epochs:
            alpha_optim.step()
        
        # weight decay loss
        loss_2 = model.weight_decay_loss(config.w_weight_decay) + model.alpha_decay_loss(config.alpha_weight_decay)
        
        alpha_optim.zero_grad()
        w_optim.zero_grad()
        loss_2.backward()
        
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)  # gradient clipping
        w_optim.step()
        alpha_optim.step()
        
        model.adjust_alphas()
        
        loss = loss_1 + loss_2

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader) - 1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch + 1, config.epochs, step, len(train_loader) - 1, losses=losses,
                    top1=top1, top5=top5))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))


if __name__ == "__main__":
    main()


