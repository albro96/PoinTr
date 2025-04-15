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
import torch.autograd.profiler as profiler
from torch.nn import functional as F
sys.path.append("/storage/share/code/01_scripts/modules/")
from general_tools.format import format_duration
import math
# from ml_tools.metrics import get_occlusion_loss

def kld_weight(config, niter):
    '''
    compute the kld weight for VAE
    '''
    start = config.kldweight.start
    target = config.kldweight.target
    ntime = config.kldweight.ntime

    _niter = niter - 10000
    if _niter > ntime:
        kld_weight = target
    elif _niter < 0:
        kld_weight = 0.
    else:
        kld_weight = target + (start - target) *  (1. + math.cos(math.pi * float(_niter) / ntime)) / 2.

    return kld_weight

def get_temp(config, niter):
    if config.get('temp') is not None:
        start = config.temp.start
        target = config.temp.target
        ntime = config.temp.ntime
        if niter > ntime:
            return target
        else:
            temp = target + (start - target) *  (1. + math.cos(math.pi * float(niter) / ntime)) / 2.
            return temp
    else:
        return 0 
    

def calc_adaptive_weight(losses, config, args):
    geom_metrics = []

    for idx, loss in enumerate(config.loss_metrics):
        if loss in ["SparseLoss", "DenseLoss"]: 
            geom_metrics.append(losses.avg()[idx])

    if (
        any([metric in Metrics.OCCLUSIONFUNCS.keys() for metric in config.loss_metrics])
        and geom_metrics
    ):
        mean_geom_loss = torch.mean(torch.tensor(geom_metrics))
        scores = torch.tensor(
            [
                config.loss.adaptive_steepness * (config.loss.adaptive_thresh - mean_geom_loss),
                config.loss.adaptive_steepness * (mean_geom_loss - config.loss.adaptive_thresh),
            ],
            device=args.device,
        )

        # Compute softmax over the two scores.
        return F.softmax(scores, dim=0)[0].item()
    else:
        ValueError("No occlusion losses in config")



def build_loss(base_model, partial, gt, config, antagonist=None, normalize=True, occl_weight=None, kwargs=None):
    losses = EasyDict()
    # create keys from losses_list
    for key in config.loss_metrics:
        losses[key] = None

    _loss = 0
    # check if partial has nan
    if 'base_model' in kwargs:
        ret = base_model(partial, **kwargs['base_model'])
    else:
        ret = base_model(partial)
    
    if config.model.NAME in ["AdaPoinTr", "PoinTr", "PCN", "DiscreteVAE"]:
        recon = ret[-1] 
        loss_func = None
        if config.loss.cd_type == 'InfoCDL2':
            loss_func = Metrics._get_info_chamfer_distancel2
        if config.model.NAME == 'DiscreteVAE':
            sparse_loss, dense_loss, klv_loss = base_model.module.get_loss(ret, gt)
            if 'KLVLoss' in config.loss_metrics:
                losses.KLVLoss = klv_loss
                _loss += klv_loss * kwargs.loss.kld_weight * config.loss.coeffs.get("KLVLoss", 1.0)
        else:
            sparse_loss, dense_loss = base_model.module.get_loss(ret=ret, gt=gt, loss_func=loss_func, config=config.loss)

    elif config.model.NAME == "CRAPCN":
        cd_loss, cra_losses = base_model.module.get_loss(
            ret=ret, gt=gt, partial=partial, config=config
        )
        sparse_loss = cra_losses[0]
        dense_loss = cra_losses[-1]

    if "SparseLoss" in config.loss_metrics:
        if config.model.NAME in ["AdaPoinTr", "PoinTr", "PCN", 'DiscreteVAE']:
            _loss += sparse_loss * config.loss.coeffs.get("SparseLoss", 1.0)
            losses.SparseLoss = sparse_loss

        elif config.model.NAME == "CRAPCN":
            if all(
                [loss in config.loss_metrics for loss in ["SparseLoss", "DenseLoss"]]
            ):
                _loss += cd_loss
                losses.SparseLoss = cd_loss
                losses.DenseLoss = cd_loss
            else:
                _loss += sparse_loss * config.loss.coeffs.get("SparseLoss", 1.0)
                losses.SparseLoss = sparse_loss

    if "DenseLoss" in config.loss_metrics:
        if config.model.NAME in ["AdaPoinTr", "PoinTr", "PCN", 'DiscreteVAE']:
            _loss += dense_loss * config.loss.coeffs.get("DenseLoss", 1.0)
            losses.DenseLoss = dense_loss

        elif config.model.NAME == "CRAPCN":
            if all(
                [loss in config.loss_metrics for loss in ["SparseLoss", "DenseLoss"]]
            ):
                pass
            else:
                _loss += dense_loss * config.loss.coeffs.get("DenseLoss", 1.0)
                losses.DenseLoss = dense_loss

    loss_metrics_filtered = [
        metric for metric in config.loss_metrics if metric in Metrics.names()
    ]
    if loss_metrics_filtered:
        metrics = Metrics.get(
            pred=recon,
            gt=gt,
            partial=None,
            antagonist=antagonist,
            metrics=loss_metrics_filtered,
            requires_grad=True,
        )

        for metric in loss_metrics_filtered:
            losses[metric] = metrics[metric]

            metric_loss = metrics[metric] * config.loss.coeffs.get(metric, 1.0)

            if config.loss.adaptive and metric in Metrics.OCCLUSIONFUNCS.keys():
                metric_loss *= occl_weight
            _loss += metric_loss

    return _loss, losses


def run_net(args, config):
    # build dataset

    train_sampler, train_dataloader = builder.dataset_builder(
        args, config, mode="train", bs=config.bs
    )
    _, val_dataloader = builder.dataset_builder(args, config, mode="val", bs=1)
    _, test_dataloader = builder.dataset_builder(
        args, config, mode="test", bs=1
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
            find_unused_parameters=False,
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
        base_model,
        optimizer,
        config,
        last_epoch=-1 if start_epoch == 1 else start_epoch - 1,
    )

# Before watching the model, do:
    if args.log_data and (not args.distributed or args.local_rank == 0):
        wandb.watch(base_model)

    epoch_time_list = []

    # base_model.zero_grad()
    occl_weights = [0.0]
    window_size = 10

    # metrics = validate(base_model, val_dataloader, start_epoch, args, config)

    for epoch in range(start_epoch, config.max_epoch + 1):

        if len(occl_weights) < window_size:
            occl_weight = 0.0
        else:
            occl_weight = torch.mean(torch.tensor(occl_weights[-window_size:])).item()

        epoch_start_time = time.time()
        epoch_allincl_start_time = time.time()

        if args.distributed:
            train_sampler.set_epoch(epoch)

        base_model.train()

        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter(config.loss_metrics)

        num_iter = 0

        n_batches = len(train_dataloader)

        # with profiler.profile(record_shapes=True, use_cuda=True) as prof:
        for idx, data in enumerate(train_dataloader):

            optimizer.zero_grad()
            data_time.update(time.time() - batch_start_time)

            if config.datasettype == 'TeethSegDataset':
                if config.dataset.return_normals:
                    partial = data[0][:, 0].to(args.device)
                    gt = data[1][:, 0].to(args.device)
                else:
                    partial = data[0].to(args.device)
                    gt = data[1].to(args.device)
                if config.dataset.return_antagonist:
                    antagonist = data[-1].to(args.device)
                else:
                    antagonist = None
            elif config.datasettype == 'SingleToothDataset':
                partial = data.to(args.device)
                gt=partial
                antagonist=None

            num_iter += 1

            if config.model_name in ['DiscreteVAE']:      
                kwargs = EasyDict({'base_model':{
                    'temperature': get_temp(config, num_iter),
                    'hard': False
                    },
                    'loss': {
                        'kld_weight': kld_weight(config, num_iter)
                    }
                })

            with torch.cuda.amp.autocast(enabled=args.use_amp_autocast):
                _loss, losses_batch = build_loss(
                    base_model=base_model,
                    partial=partial,
                    gt=gt,
                    config=config,
                    antagonist=antagonist,
                    occl_weight=occl_weight,
                    kwargs=kwargs,
                )

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
                for loss in losses_batch.keys():
                    losses_batch[loss] = dist_utils.reduce_tensor(
                        losses_batch[loss], args
                    )
                losses_batch_list = [losses_batch[loss] for loss in config.loss_metrics]
            else:
                # print("losses_batch", losses_batch)
                losses_batch_list = [
                    losses_batch[loss].item() for loss in config.loss_metrics
                ]

            losses.update(losses_batch_list)

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
            del _loss
            torch.cuda.empty_cache()

        # print(prof.key_averages().table(sort_by="cuda_time_total"))

        if args.sweep and args.log_data and (not args.distributed or args.local_rank == 0):
           print(
                f'\rAgent: {wandb.run.id} Epoch [{epoch}/{config.max_epoch}] Losses = {[f"{l:.4f}"  for l in losses.val()]} lr = {optimizer.param_groups[0]["lr"]:.6f}'
            )

        if config.scheduler.type == "GradualWarmup":
            if n_itr < config.scheduler.kwargs_2.total_epoch:
                scheduler.step(epoch)
        if config.scheduler.type != 'function':
            if isinstance(scheduler, list):
                for item in scheduler:
                    item.step(epoch)
            else:
                scheduler.step(epoch)
        epoch_end_time = time.time()

        if not args.distributed or args.local_rank == 0:
            if epoch % args.val_freq == 0:
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

            if args.save_checkpoints and not args.save_only_best and args.log_data:
                builder.save_checkpoint(
                    base_model, optimizer, epoch, metrics, best_metrics, f"ckpt-last-{wandb.run.name}", args
                )
                # save every 100 epoch
                if epoch % 100 == 0:
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

        if config.loss.adaptive:
            occl_weights.append(calc_adaptive_weight(losses, config, args))

# After training an epoch, log metrics only on master.
        if args.log_data and (not args.distributed or args.local_rank == 0):
            log_dict = EasyDict()
            log_dict.epoch = epoch
            if config.loss.adaptive:
                log_dict.occl_weight = occl_weight
            for idx, loss in enumerate(config.loss_metrics):
                log_dict[f"train/{loss}"] = losses.avg()[idx]

            wandb.log(log_dict, step=epoch)

        epoch_allincl_end_time = time.time()

        epoch_time_list.append(epoch_allincl_end_time - epoch_allincl_start_time)

        mean_epoch_time = sum(epoch_time_list) / len(epoch_time_list)

        est_time = mean_epoch_time * (config.max_epoch - epoch + 1)

        print(
            f'[Training] EPOCH: {epoch}/{config.max_epoch} EpochTime = {format_duration(epoch_time_list[-1])} Remaining Time = {format_duration(est_time)} Losses = {["%.4f" % l for l in losses.avg()]} \n'
        )


def validate(base_model, val_dataloader, epoch, args, config, logger=None):
    base_model.eval()  # set model to eval mode

    val_metrics = {metric: [] for metric in config.val_metrics}
    if config.model.NAME in ['DiscreteVAE']:
        kwargs = EasyDict({'base_model':{
                        'hard': True,
                        }
                    })
    else:
        kwargs = EasyDict()

    with torch.no_grad():
        for idx, data in enumerate(val_dataloader):
            if config.datasettype == 'TeethSegDataset':
                if config.dataset.return_normals:
                    partial = data[0][:, 0].to(args.device)
                    gt = data[1][:, 0].to(args.device)
                else:
                    partial = data[0].to(args.device)
                    gt = data[1].to(args.device)

                if config.dataset.return_antagonist:
                    antagonist = data[-1].to(args.device)
                else:
                    antagonist = None
            elif config.datasettype == 'SingleToothDataset':
                partial = data.to(args.device)
                gt=partial
                antagonist=None

            with torch.cuda.amp.autocast(enabled=args.use_amp_autocast):
                if 'base_model' in kwargs:
                    ret = base_model(partial, **kwargs['base_model'])
                else:
                    ret = base_model(partial)

            coarse_points = ret[-2]
            dense_points = ret[-1]

            if (
                val_dataloader.dataset.patient == "0U1LI1CB"
                or val_dataloader.dataset.patient == "0538") and args.log_data:
                print('Logging PCDs')
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
                        ),
                        # f"val/pcd/coarse/{val_dataloader.dataset.tooth}": wandb.Object3D(
                        #     {
                        #         "type": "lidar/beta",
                        #         "points": coarse_points[0].detach().cpu().numpy(),
                        #     }
                        # ),
                        # f"val/pcd/full-dense/{val_dataloader.dataset.tooth}": wandb.Object3D(
                        #     {
                        #         "type": "lidar/beta",
                        #         "points": full_dense[0].detach().cpu().numpy(),
                        #     }
                        # ),
                        f"val/pcd/gt/{val_dataloader.dataset.tooth}": wandb.Object3D(
                            {
                                "type": "lidar/beta",
                                "points": gt[0].detach().cpu().numpy(),
                            }
                        ),
                        # f"val/pcd/partial/{val_dataloader.dataset.tooth}": wandb.Object3D(
                        #     {
                        #         "type": "lidar/beta",
                        #         "points": partial[0].detach().cpu().numpy(),
                        #     }
                        # ),
                    },
                    step=epoch,
                )

            _metrics = Metrics.get(
                pred=dense_points,
                gt=gt,
                partial=partial,
                antagonist=antagonist,
                metrics=config.val_metrics,
                requires_grad=False,
            )

            # val_dataloader was set to non distributed
            # if args.distributed:
            #     for metric, value in _metrics.items():
            #         _metrics[metric] = dist_utils.reduce_tensor(value, args).item()
            # else:
            for metric, value in _metrics.items():
                _metrics[metric] = value.item()
            
            for metric in val_metrics:
                val_metrics[metric].append(_metrics[metric])


        # if args.distributed:
        #     torch.cuda.synchronize()

    print("============================ VAL RESULTS ============================")
    print(f"Epoch: {epoch}")
    print('Metric: MEAN/MEDIAN')
    log_dict = {}  # {'val/epoch': epoch}

    if any(metric in val_metrics for metric in config.occlusion_metrics):
        filter_arr = torch.tensor(val_metrics['ClusterPosLoss']) != args.no_occlusion_val
        log_dict['val/num_success'] = torch.sum(filter_arr).item()
        log_dict['val/num_failed'] = len(val_metrics['ClusterPosLoss']) - log_dict['val/num_success']
        print(f"Num success: {log_dict['val/num_success']}")
        print(f"Num failed: {log_dict['val/num_failed']}")

    mean_val_metrics = []
    for metric, values in val_metrics.items():
        values = torch.tensor(values)
        mean = torch.mean(values)
        mean_val_metrics.append(mean)
        median = torch.median(values)
        log_dict[f"val/{metric}"] = mean
        log_dict[f"val/{metric}-median"] = median
        print(f"{metric}: {mean:.6f}/{median:.6f}")

    if args.log_data:
        wandb.log(log_dict, step=epoch)

    return Metrics(
        metric_name=config.consider_metric,
        values=mean_val_metrics,
        metrics=config.val_metrics,
    )

def test_net(args, config):
    import trimesh
    from tqdm import tqdm
    from hashlib import sha256
    import pickle
    from datasets.TeethSegDataset import fps_with_normals
    from pytorch3d.ops import sample_farthest_points as fps

    print(f"Tester start ... on device: {args.device} ")

    data_dir = "/storage/share/data/3d-datasets/studscan/data/"
    cache_dir = "/storage/share/data/3d-datasets/studscan/cache/"
    
    cache_dict = {
        "num_points_corr": config.dataset.num_points_corr,
        "num_points_gt": config.dataset.num_points_gt,
        "num_points_corr_anta": config.dataset.num_points_corr_anta,
        'return_normals': config.dataset.return_normals,
        'return_antagonist': config.dataset.return_antagonist,
    }

    # create hash
    cache_hash = sha256(
                json.dumps(cache_dict, sort_keys=True).encode()
            ).hexdigest()[:8]

    cache_file_path = os.path.join(cache_dir, f"test_dataloader_{cache_hash}.pkl")

    if os.path.exists(cache_file_path):
        with open(cache_file_path, "rb") as f:
            test_dataloader = pickle.load(f)
        print("Loaded test dataloader from cache")
        # return test_dataloader
    else:
        patients = os.listdir(data_dir)
        test_dataloader = []
        for patient in tqdm(patients):
            patient_dir = os.path.join(data_dir, patient)

            lower_path = os.path.join(patient_dir, f'{patient}_lower_corr.stl')
            upper_path = os.path.join(patient_dir, f'{patient}_upper.stl')
            gt_path = os.path.join(patient_dir, f'{patient}_gt.stl')
            anta_path = os.path.join(patient_dir, f'{patient}_anta.stl')

            lower_mesh = trimesh.load_mesh(lower_path, process=False)
            lower = torch.tensor(lower_mesh.vertices, device=args.device).unsqueeze(0)
            upper_mesh = trimesh.load_mesh(upper_path, process=False)
            upper = torch.tensor(upper_mesh.vertices, device=args.device).unsqueeze(0)
            anta_mesh = trimesh.load_mesh(anta_path, process=False)
            anta = torch.tensor(anta_mesh.vertices, device=args.device).unsqueeze(0)
            anta_norm = torch.tensor(anta_mesh.vertex_normals, device=args.device).unsqueeze(0)

            gt_mesh= trimesh.load_mesh(gt_path, process=False)
            gt = torch.tensor(gt_mesh.vertices, device=args.device).unsqueeze(0)
            
            if config.dataset.return_normals:
                gt_norm = torch.tensor(gt_mesh.vertex_normals, device=args.device).unsqueeze(0).unsqueeze(0)
                gt = torch.cat([gt.unsqueeze(0), gt_norm], dim=1)
                gt = fps_with_normals(gt, n_points=config.dataset.num_points_gt)
            else:
                gt, _ = fps(gt, K=config.dataset.num_points_gt)

            n_pts_corr_half = int(config.dataset.num_points_corr/2)
            if config.dataset.return_normals:
                lower_norm = torch.tensor(lower_mesh.vertex_normals, device=args.device).unsqueeze(0).unsqueeze(0)
                lower = torch.cat([lower.unsqueeze(0), lower_norm], dim=1)
                lower = fps_with_normals(lower, n_points=n_pts_corr_half)

                upper_norm = torch.tensor(upper_mesh.vertex_normals, device=args.device).unsqueeze(0).unsqueeze(0)
                upper = torch.cat([upper.unsqueeze(0), upper_norm], dim=1)
                upper = fps_with_normals(upper, n_points=n_pts_corr_half)
            else:
                lower, _ = fps(lower, K=n_pts_corr_half)
                upper, _ = fps(upper, K=n_pts_corr_half)

            corr = torch.cat([lower, upper], dim=-2)

            anta = torch.cat([anta.unsqueeze(0), anta_norm.unsqueeze(0)], dim=1)
            anta = fps_with_normals(anta, n_points=config.dataset.num_points_corr_anta)

            corr = corr.float()
            gt = gt.float()
            anta = anta.float()

            test_dataloader.append((corr, gt, anta))

        os.makedirs(cache_dir, exist_ok=True)
        # save test_dataloader as pkl
        with open(cache_file_path, "wb") as f:
            pickle.dump(test_dataloader, f)

    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpt_path)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config)



def test(base_model, test_dataloader, args, config):
    from pcd_tools.data_processing import torch_to_o3d
    from tqdm import tqdm

    save_dir = os.path.join(args.experiment_path, 'test_data', os.path.basename(args.ckpt_path).split('.')[0], 'pcd')

    base_model.eval()  # set model to eval mode

    test_metrics = {metric: [] for metric in config.val_metrics}

    if config.model.NAME in ['DiscreteVAE']:
        kwargs = EasyDict({'base_model':{
                        'hard': True,
                        }
                    })
    else:
        kwargs = EasyDict()


    with torch.no_grad():
        for idx, data in tqdm(enumerate(test_dataloader)):

            if config.datasettype == 'TeethSegDataset':
                if config.dataset.return_normals:
                    partial = data[0][:, 0].to(args.device)
                    gt = data[1][:, 0].to(args.device)
                else:
                    partial = data[0].to(args.device)
                    gt = data[1].to(args.device)

                if config.dataset.return_antagonist:
                    antagonist = data[-1].to(args.device)
                else:
                    antagonist = None
            elif config.datasettype == 'SingleToothDataset':
                partial = data.to(args.device)
                gt=partial
                antagonist=None

            with torch.cuda.amp.autocast(enabled=args.use_amp_autocast):
                if 'base_model' in kwargs:
                    ret = base_model(partial, **kwargs['base_model'])
                else:
                    ret = base_model(partial)

            coarse_points = ret[-2]
            dense_points = ret[-1]

            if args.log_data:
                print('Logging PCDs')
                if not config.model.NAME == "CRAPCN":
                    full_dense = torch.cat([partial, dense_points], dim=1)
                else:
                    full_dense = dense_points

            if args.log_testdata:
                os.makedirs(save_dir, exist_ok=True)
                torch_to_o3d(dense_points, os.path.join(save_dir, f'{idx:02d}_dense.pcd'))
                torch_to_o3d(partial, os.path.join(save_dir, f'{idx:02d}_partial.pcd'))
                if not config.model.NAME in ['DiscreteVAE']:
                    torch_to_o3d(gt, os.path.join(save_dir, f'{idx:02d}_gt.pcd'))
                    torch_to_o3d(antagonist, os.path.join(save_dir, f'{idx:02d}_anta.pcd'))
                
            _metrics = Metrics.get(
                pred=dense_points,
                gt=gt,
                partial=partial,
                antagonist=antagonist,
                metrics=config.val_metrics,
                requires_grad=False,
            )

            for metric, value in _metrics.items():
                _metrics[metric] = value.item()
        

            for metric in test_metrics:
                test_metrics[metric].append(_metrics[metric])

    print("============================ TEST RESULTS ============================")
    if args.log_testdata:
        import pickle
        metric_dir = os.path.dirname(save_dir)
        print(f"Test data saved to {metric_dir}")
        with open(os.path.join(metric_dir, 'test_metrics.pkl'), 'wb') as f:
            pickle.dump(test_metrics, f)
        # calc mean, median, etc and save to json
        metrics_dict = EasyDict({'all': {}, 'cleaned': {}})
        filter_arr = None
        if any(metric in test_metrics for metric in config.occlusion_metrics):
            filter_arr = torch.tensor(test_metrics['ClusterPosLoss']) != args.no_occlusion_val
            metrics_dict['all']['num_success'] = torch.sum(filter_arr).item()
            metrics_dict['all']['num_failed'] = len(test_metrics['ClusterPosLoss']) - metrics_dict['all']['num_success']

        for calctype in ['all', 'cleaned']:
            if filter_arr is None:
                continue
            
            for metric, value in test_metrics.items():
                value = torch.tensor(value)
                if calctype == 'cleaned':
                    value = value[filter_arr]

                metrics_dict[calctype][metric] = {
                    'mean': torch.mean(value).item(),
                    'median': torch.median(value).item(),
                    'min': torch.min(value).item(),
                    'max': torch.max(value).item(),
                    '75th': torch.quantile(value, 0.75).item(),
                    '25th': torch.quantile(value, 0.25).item(),
                    '95th': torch.quantile(value, 0.95).item(),
                    '99th': torch.quantile(value, 0.99).item(),
                }

        with open(os.path.join(metric_dir, 'test_metrics.json'), 'w') as f:
            json.dump(metrics_dict, f, indent=4)
            
            
    for metric, value in test_metrics.items():
        # count occurences of args.no_occlusion_val
        if metric in ['OcclusionLoss', 'ClusterDistLoss', 'ClusterNumLoss', 'ClusterPosLoss', 'PenetrationLoss']:
            ctr = value.count(args.no_occlusion_val)
        print(f"{metric}: {torch.median(torch.tensor(value)):.6f}")
    print(f"Failed Occlusions: {ctr} / {len(test_metrics['OcclusionLoss'])}")

    # log_dict = {}  # {'val/epoch': epoch}
    # for metric, value in zip(test_metrics.items, test_metrics.avg()):
    #     log_dict[f"test/{metric}"] = value
    #     print(f"{metric}: {value:.6f}")

    # return Metrics(
    #     metric_name=config.consider_metric,
    #     values=test_metrics.avg(),
    #     metrics=config.test_metrics,
    # )
