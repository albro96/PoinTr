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

sys.path.append("/storage/share/code/01_scripts/modules/")
from general_tools.format import format_duration


def run_net(args, config):
    # build dataset
    train_sampler, train_dataloader = builder.dataset_builder(
        args, config.dataset, mode="train", bs=config.bs
    )
    _, val_dataloader = builder.dataset_builder(args, config.dataset, mode="val", bs=1)
    _, test_dataloader = builder.dataset_builder(
        args, config.dataset, mode="test", bs=1
    )

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
            print("Using Synchronized BatchNorm ...")
        base_model = nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[args.local_rank % torch.cuda.device_count()],
            find_unused_parameters=True,
        )
        print("Using Distributed Data parallel ...")
    else:  # elif args.num_gpus > 1:
        print("Using Data parallel ...")
        base_model = nn.DataParallel(base_model).to(args.device)

    # optimizer & scheduler
    optimizer = builder.build_optimizer(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args)
    scheduler = builder.build_scheduler(
        base_model, optimizer, config, last_epoch=start_epoch - 1
    )

    if args.log_data:
        wandb.watch(base_model)

    # trainval
    # training
    epoch_time_list = []

    # base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        epoch_start_time = time.time()
        epoch_allincl_start_time = time.time()

        if args.distributed:
            train_sampler.set_epoch(epoch)

        base_model.train()

        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(["SparseLoss", "DenseLoss"])

        num_iter = 0

        # base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        for idx, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            # print(train_dataloader.dataset.patient, train_dataloader.dataset.tooth)
            data_time.update(time.time() - batch_start_time)
            # npoints = config.dataset.train._base_.N_POINTS
            # dataset_name = config.dataset.train._base_.NAME

            partial = data[0].to(args.device)
            gt = data[1].to(args.device)

            num_iter += 1

            with torch.cuda.amp.autocast(enabled=args.use_amp_autocast):
                # check if partial has nan
                ret = base_model(partial)

                if config.model.NAME in ["AdaPoinTr", "PoinTr", "PCN"]:
                    sparse_loss, dense_loss = base_model.module.get_loss(
                        ret=ret, gt=gt, epoch=epoch
                    )
                    if config.model.NAME == "AdaPoinTr":
                        _loss = (
                            sparse_loss + dense_loss
                        )  # denseloss coeff is applied in model for adapointr
                    else:
                        _loss = sparse_loss + config.dense_loss_coeff * dense_loss

                elif config.model.NAME == "CRAPCN":
                    _loss, cra_losses = base_model.module.get_loss(
                        ret=ret, gt=gt, partial=partial, config=config
                    )
                    sparse_loss = cra_losses[0]
                    dense_loss = cra_losses[-1]

            _loss.backward()

            # forward
            if num_iter == config.step_per_update:

                torch.nn.utils.clip_grad_norm_(
                    base_model.parameters(),
                    max_norm=getattr(config, "grad_norm_clip", 0),
                    norm_type=2,
                    error_if_nonfinite=True,
                )
                # print("grad norm clip", getattr(config, "grad_norm_clip", 0))
                num_iter = 0
                optimizer.step()
                # base_model.zero_grad()

            if args.distributed:
                sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                dense_loss = dist_utils.reduce_tensor(dense_loss, args)
                losses.update([sparse_loss.item(), dense_loss.item()])
            else:
                losses.update([sparse_loss.item(), dense_loss.item()])

            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            # epoch progress
            if not args.sweep:
                print(
                    f'\r[Epoch {epoch}/{config.max_epoch}][Batch {idx}/{n_batches}] BatchTime = {format_duration(batch_time.val(-1))} Loss = {_loss:.4f} lr = {optimizer.param_groups[0]["lr"]:.6f}',
                    end="\r",
                )

        if args.sweep and args.log_data:
            print(
                f'\rAgent: {wandb.run.id} Epoch [{epoch}/{config.max_epoch}] Losses = {[f"{l:.4f}"  for l in losses.val()]} lr = {optimizer.param_groups[0]["lr"]:.6f}'
            )

        if config.scheduler.type == "GradualWarmup":
            if n_itr < config.scheduler.kwargs_2.total_epoch:
                scheduler.step()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()
        epoch_end_time = time.time()

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics = validate(base_model, val_dataloader, epoch, args, config)

            # Save ckeckpoints
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                if args.save_checkpoints and args.log_data:
                    builder.save_checkpoint(
                        base_model,
                        optimizer,
                        epoch,
                        metrics,
                        best_metrics,
                        f"ckpt-best-{wandb.run.name}",
                        args,
                    )

        if args.test_freq is not None and epoch % args.test_freq == 0:
            metrics = test(base_model, test_dataloader, args, config, epoch=epoch)

        if args.save_checkpoints and not args.save_only_best and args.log_data:
            builder.save_checkpoint(
                base_model, optimizer, epoch, metrics, best_metrics, "ckpt-last", args
            )
            # save every 100 epoch
            if epoch % 100 == 0 and epoch != 0:
                builder.save_checkpoint(
                    base_model,
                    optimizer,
                    epoch,
                    metrics,
                    best_metrics,
                    f"ckpt-epoch-{epoch:03d}-{wandb.run.name}",
                    args,
                )

            if (config.max_epoch - epoch) < 2:
                builder.save_checkpoint(
                    base_model,
                    optimizer,
                    epoch,
                    metrics,
                    best_metrics,
                    f"ckpt-epoch-{epoch:03d}-{wandb.run.name}",
                    args,
                )

        if args.log_data:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/sparse_loss": losses.avg()[0],
                    "train/dense_loss": losses.avg()[1],
                },
                step=epoch,
            )

        epoch_allincl_end_time = time.time()

        epoch_time_list.append(epoch_allincl_end_time - epoch_allincl_start_time)

        mean_epoch_time = sum(epoch_time_list) / len(epoch_time_list)

        est_time = mean_epoch_time * (config.max_epoch - epoch + 1)

        print(
            f'[Training] EPOCH: {epoch}/{config.max_epoch} EpochTime = {format_duration(epoch_time_list[-1])} Remaining Time = {format_duration(est_time)} Losses = {["%.4f" % l for l in losses.avg()]} \n'
        )


def validate(base_model, val_dataloader, epoch, args, config, logger=None):
    # print(f"\n[VALIDATION] Start validating epoch {epoch}")
    base_model.eval()  # set model to eval mode

    val_losses = AverageMeter(
        ["SparseLossL1", "SparseLossL2", "DenseLossL1", "DenseLossL2"]
    )
    val_metrics = AverageMeter(Metrics.names())

    with torch.no_grad():
        for idx, data in enumerate(val_dataloader):

            partial = data[0].to(args.device)
            gt = data[1].to(args.device)

            with torch.cuda.amp.autocast(enabled=args.use_amp_autocast):
                ret = base_model(partial)

            coarse_points = ret[0]
            dense_points = ret[-1]

            if val_dataloader.dataset.patient == "0U1LI1CB" and args.log_data:
                if not config.model.NAME == "CRAPCN":
                    full_dense = torch.cat([partial, dense_points], dim=1)
                else:
                    full_dense = dense_points

                wandb.log(
                    {
                        f"val/pcd/dense/{val_dataloader.dataset.tooth}": wandb.Object3D(
                            {
                                "type": "lidar/beta",
                                "points": dense_points[0].detach().cpu().numpy(),
                            }
                        )
                    },
                    step=epoch,
                )
                wandb.log(
                    {
                        f"val/pcd/coarse/{val_dataloader.dataset.tooth}": wandb.Object3D(
                            {
                                "type": "lidar/beta",
                                "points": coarse_points[0].detach().cpu().numpy(),
                            }
                        )
                    },
                    step=epoch,
                )
                wandb.log(
                    {
                        f"val/pcd/full-dense/{val_dataloader.dataset.tooth}": wandb.Object3D(
                            {
                                "type": "lidar/beta",
                                "points": full_dense[0].detach().cpu().numpy(),
                            }
                        )
                    },
                    step=epoch,
                )
                wandb.log(
                    {
                        f"val/pcd/gt/{val_dataloader.dataset.tooth}": wandb.Object3D(
                            {
                                "type": "lidar/beta",
                                "points": gt[0].detach().cpu().numpy(),
                            }
                        )
                    },
                    step=epoch,
                )
                wandb.log(
                    {
                        f"val/pcd/partial/{val_dataloader.dataset.tooth}": wandb.Object3D(
                            {
                                "type": "lidar/beta",
                                "points": partial[0].detach().cpu().numpy(),
                            }
                        )
                    },
                    step=epoch,
                )

            # losses = {}
            # for cd_norm in [1, 2]:
            #     if config.model.NAME in ["AdaPoinTr", "PoinTr", "PCN"]:
            #         losses[f"coarse-L{cd_norm}"], losses[f"dense-L{cd_norm}"] = (
            #             base_model.module.get_loss(
            #                 ret=ret, gt=gt, epoch=epoch, cd_norm=cd_norm
            #             )
            #         )
            #     elif config.model.NAME == "CRAPCN":
            #         _, losses = base_model.module.get_loss(
            #             ret=ret, gt=gt, partial=partial, config=config, cd_norm=cd_norm
            #         )
            #         losses[f"coarse-L{cd_norm}"] = losses[0]
            #         losses[f"dense-L{cd_norm}"] = losses[-1]

            _metrics = Metrics.get(pred=dense_points, gt=gt, partial=partial)

            if args.distributed:
                _metrics = [
                    dist_utils.reduce_tensor(_metric, args).item()
                    for _metric in _metrics
                ]
            else:
                _metrics = [_metric.item() for _metric in _metrics]

            val_metrics.update(_metrics)

        if args.distributed:
            torch.cuda.synchronize()

    print("============================ VAL RESULTS ============================")
    print(f"Epoch: {epoch}")
    log_dict = {}  # {'val/epoch': epoch}
    for metric, value in zip(val_metrics.items, val_metrics.avg()):
        log_dict[f"val/{metric}"] = value
        print(f"{metric}: {value:.6f}")

    if args.log_data:
        wandb.log(log_dict, step=epoch)

    return Metrics(config.consider_metric, val_metrics.avg())


crop_ratio = {"easy": 1 / 4, "median": 1 / 2, "hard": 3 / 4}


def test_net(args, config):
    # logger = get_logger(args.log_name)
    print("Tester start ... ")
    _, test_dataloader = builder.dataset_builder(args, config.dataset, mode="test")

    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config)


def test(base_model, test_dataloader, args, config, logger=None, epoch=None):

    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(
        ["SparseLossL1", "SparseLossL2", "DenseLossL1", "DenseLossL2"]
    )
    test_metrics = AverageMeter(Metrics.names())
    # category_metrics = dict()
    n_samples = len(test_dataloader)  # bs is 1

    with torch.no_grad():
        for data in test_dataloader:

            partial = data[0].to(args.device)
            gt = data[1].to(args.device)

            with torch.cuda.amp.autocast(enabled=args.use_amp_autocast):
                ret = base_model(partial)
            coarse_points = ret[0]
            dense_points = ret[-1]

            sparse_loss_l1 = chamfer_distance(coarse_points, gt, norm=1)[0]
            sparse_loss_l2 = chamfer_distance(coarse_points, gt, norm=2)[0]
            dense_loss_l1 = chamfer_distance(dense_points, gt, norm=1)[0]
            dense_loss_l2 = chamfer_distance(dense_points, gt, norm=2)[0]

            test_losses.update(
                [
                    sparse_loss_l1.item(),
                    sparse_loss_l2.item(),
                    dense_loss_l1.item(),
                    dense_loss_l2.item(),
                ]
            )

            _metrics = Metrics.get(pred=dense_points, gt=gt, partial=partial)
            test_metrics.update(_metrics)

    print("============================ TEST RESULTS ============================")
    log_dict = {}  # {'test/epoch': epoch}
    for metric, value in zip(test_metrics.items, test_metrics.avg()):
        log_dict[f"test/{metric}"] = value
        print(f"{metric}: {value:.6f}")
    if args.log_data:
        wandb.log(log_dict, step=epoch)

    return
