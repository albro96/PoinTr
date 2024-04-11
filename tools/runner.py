import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
import sys
import wandb
from easydict import EasyDict
import pytorch3d as p3d
from pytorch3d.loss import chamfer_distance

sys.path.append('/storage/share/code/01_scripts/modules/')
from general_tools.format import format_duration

def run_net(args, config):
    # build dataset
    train_sampler, train_dataloader = builder.dataset_builder(args, config.dataset, mode='train', bs=config.bs)
    _, val_dataloader = builder.dataset_builder(args, config.dataset, mode='val', bs=1)
    _, test_dataloader = builder.dataset_builder(args, config.dataset, mode='test', bs=1)

    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)
    
    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print('Using Synchronized BatchNorm ...')
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print('Using Distributed Data parallel ...' )
    else:
        print('Using Data parallel ...' )
        base_model = nn.DataParallel(base_model).to(args.device)

    # optimizer & scheduler
    optimizer = builder.build_optimizer(base_model, config)


    if args.resume:
        builder.resume_optimizer(optimizer, args)
    scheduler = builder.build_scheduler(base_model, optimizer, config, last_epoch=start_epoch-1)

    wandb.watch(base_model)

    # trainval
    # training
    epoch_time_list = []

    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        epoch_start_time = time.time()
        epoch_allincl_start_time = time.time()
    
        if args.distributed:
            train_sampler.set_epoch(epoch)

        base_model.train()

        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['SparseLoss', 'DenseLoss'])

        num_iter = 0

        # base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        for idx, data in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            # npoints = config.dataset.train._base_.N_POINTS
            # dataset_name = config.dataset.train._base_.NAME

            partial = data[0].to(args.device)
            gt = data[1].to(args.device)

            num_iter += 1
           
            with torch.cuda.amp.autocast(enabled=args.use_amp_autocast):
                ret = base_model(partial)
                sparse_loss, dense_loss = base_model.module.get_loss(ret, gt, epoch)    
            
            # denseloss coeff is applied in model for adapointr
            if config.model.NAME == 'AdaPoinTr':
                _loss = sparse_loss + dense_loss
            else:
                _loss = sparse_loss + config.dense_loss_coeff*dense_loss 
            
            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(config, 'grad_norm_clip', 10), norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                dense_loss = dist_utils.reduce_tensor(dense_loss, args)
                losses.update([sparse_loss.item() * 1, dense_loss.item() * 1])
            else:
                losses.update([sparse_loss.item() * 1, dense_loss.item() * 1])

            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            # epoch progress
            if not args.sweep:
                print(f'\r[Epoch {epoch}/{config.max_epoch}][Batch {idx}/{n_batches}] BatchTime = {format_duration(batch_time.val(-1))} Losses = {[f"{l:.4f}"  for l in losses.val()]} lr = {optimizer.param_groups[0]["lr"]:.6f}', end="\r")

        if args.sweep:
            print(f'\rAgent: {wandb.run.id} Epoch [{epoch}/{config.max_epoch}] Losses = {[f"{l:.4f}"  for l in losses.val()]} lr = {optimizer.param_groups[0]["lr"]:.6f}')

            if config.scheduler.type == 'GradualWarmup':
                if n_itr < config.scheduler.kwargs_2.total_epoch:
                    scheduler.step()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()
        epoch_end_time = time.time()
        
        if epoch % args.val_freq == 0:
            # Validate the current model
            metrics = validate(base_model, val_dataloader, epoch, args, config)

            # Save ckeckpoints
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                if args.save_checkpoints:
                    builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-best-{wandb.run.name}', args)

        if args.test_freq is not None and epoch % args.test_freq == 0:
            metrics = test(base_model, test_dataloader, args, config, epoch=epoch)

            
        if args.save_checkpoints and not args.save_only_best:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args)  
            # save every 100 epoch
            if epoch % 100 == 0 and epoch != 0:
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}-{wandb.run.name}', args)  

            if (config.max_epoch - epoch) < 2:
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}-{wandb.run.name}', args)   

        epoch_allincl_end_time = time.time()

        epoch_time_list.append(epoch_allincl_end_time - epoch_allincl_start_time)

        mean_epoch_time = sum(epoch_time_list) / len(epoch_time_list)

        est_time = mean_epoch_time * (config.max_epoch - epoch + 1)
        
        print(f'[Training] EPOCH: {epoch}/{config.max_epoch} EpochTime = {format_duration(epoch_time_list[-1])} Remaining Time = {format_duration(est_time)} Losses = {["%.4f" % l for l in losses.avg()]} \n' )

        wandb.log(
            {
                "train/epoch": epoch,
                "train/sparse_loss": losses.avg()[0],
                "train/dense_loss": losses.avg()[1], 
            })
        

def validate(base_model, val_dataloader, epoch, args, config, logger = None):
    # print(f"\n[VALIDATION] Start validating epoch {epoch}")
    base_model.eval()  # set model to eval mode

    val_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    val_metrics = AverageMeter(Metrics.names())

    with torch.no_grad():
        for idx, data in enumerate(val_dataloader):

            partial = data[0].to(args.device)
            gt = data[1].to(args.device)

            with torch.cuda.amp.autocast(enabled=args.use_amp_autocast):
                ret = base_model(partial)
            coarse_points = ret[0]
            dense_points = ret[-1]

            if idx == 0:
                full_dense = torch.cat([partial, dense_points], dim=1)
                wandb.log({"val/dense": wandb.Object3D({"type": "lidar/beta","points": dense_points[0].detach().cpu().numpy(),})})
                wandb.log({"val/coarse": wandb.Object3D({"type": "lidar/beta","points": coarse_points[0].detach().cpu().numpy(),})})
                wandb.log({"val/full-dense": wandb.Object3D({"type": "lidar/beta","points": full_dense[0].detach().cpu().numpy(),})})

                if not args.gt_partial_saved:
                    # save gt and partial only once
                    wandb.log({"val/gt": wandb.Object3D({"type": "lidar/beta","points": gt[0].detach().cpu().numpy(),})})
                    wandb.log({"val/partial": wandb.Object3D({"type": "lidar/beta","points": partial[0].detach().cpu().numpy(),})})
                    args.gt_partial_saved = True
                
            sparse_loss_l1 =  chamfer_distance(coarse_points, gt, norm=1)[0]
            sparse_loss_l2 =  chamfer_distance(coarse_points, gt, norm=2)[0]
            dense_loss_l1 =  chamfer_distance(dense_points, gt, norm=1)[0]
            dense_loss_l2 =  chamfer_distance(dense_points, gt, norm=2)[0]

            if args.distributed:
                sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

            val_losses.update([sparse_loss_l1.item() * 1, sparse_loss_l2.item() * 1, dense_loss_l1.item() * 1, dense_loss_l2.item() * 1])

            _metrics = Metrics.get(pred=dense_points, gt=gt, partial=partial)

            if args.distributed:
                _metrics = [dist_utils.reduce_tensor(_metric, args).item() for _metric in _metrics]
            else:
                _metrics = [_metric.item() for _metric in _metrics]

            val_metrics.update(_metrics) 

        # print(f'[Validation] EPOCH: {epoch}  Metrics = {[f"{m:.4f}" for m in val_metrics.avg()]}')
        
        if args.distributed:
            torch.cuda.synchronize()
    
    print('============================ VAL RESULTS ============================')
    print(f'Epoch: {epoch}')
    log_dict = {'val/epoch': epoch}
    for metric, value in zip(val_metrics.items, val_metrics.avg()):
        log_dict[f"val/{metric}"] = value
        print(f'{metric}: {value:.6f}')

    wandb.log(log_dict)

    return Metrics(config.consider_metric, val_metrics.avg())


crop_ratio = {
    'easy': 1/4,
    'median' :1/2,
    'hard':3/4
}

def test_net(args, config):
    # logger = get_logger(args.log_name)
    print('Tester start ... ')
    _, test_dataloader = builder.dataset_builder(args, config.dataset, mode='test')
 
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config)

def test(base_model, test_dataloader, args, config, logger = None, epoch=None):

    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    # category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    with torch.no_grad():
        for data in test_dataloader:

            partial = data[0].to(args.device)
            gt = data[1].to(args.device)

            with torch.cuda.amp.autocast(enabled=args.use_amp_autocast):
                ret = base_model(partial)
            coarse_points = ret[0]
            dense_points = ret[-1]

            sparse_loss_l1 =  chamfer_distance(coarse_points, gt, norm=1)[0]
            sparse_loss_l2 =  chamfer_distance(coarse_points, gt, norm=2)[0]
            dense_loss_l1 =  chamfer_distance(dense_points, gt, norm=1)[0]
            dense_loss_l2 =  chamfer_distance(dense_points, gt, norm=2)[0]

            test_losses.update([sparse_loss_l1.item() * 1, sparse_loss_l2.item() * 1, dense_loss_l1.item() * 1, dense_loss_l2.item() * 1])

            _metrics = Metrics.get(pred=dense_points, gt=gt, partial=partial)
            test_metrics.update(_metrics)

    print('============================ TEST RESULTS ============================')
    log_dict = {'test/epoch': epoch}
    for metric, value in zip(test_metrics.items, test_metrics.avg()):
        log_dict[f"test/{metric}"] = value
        print(f'{metric}: {value:.6f}')
    wandb.log(log_dict)

    return 
