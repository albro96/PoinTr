import time
import os
import torch
import sys
from easydict import EasyDict
import os.path as op
from pprint import pprint

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(BASE_DIR, "../"))
sys.path.append("/storage/share/code/01_scripts/modules/")

from os_tools.import_dir_path import import_dir_path

from datasets.TeethSegDataset import TeethSegDataset

# ---------------------------------------- #
# ----------------- Main ----------------- #
# ---------------------------------------- #
pada = import_dir_path()

data_config = EasyDict(
    {
        "num_points_gt": 1024,
        "num_points_corr": 1024,  # 2048 4096 8192 16384
        "num_points_corr_type": "single",
        "tooth_range": {
            "corr": "31-37",
            "gt": [36],
            "jaw": "lower",
            "quadrants": "all",
        },
        "gt_type": "single",
        "data_type": "npy",
        "samplingmethod": "fps",
        "downsample_steps": 2,
        "splits": {"train": 0.85, "val": 0.1, "test": 0.05},
        "enable_cache": True,
        "create_cache_file": True,
        "overwrite_cache_file": False,
        "cache_dir": op.join(
            pada.base_dir, "nobackup", "data", "3DTeethSeg22", "cache"
        ),
    }
)


train_set = TeethSegDataset(
    mode="train",
    **data_config,
    device=torch.device("cuda:0"),
)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=1, shuffle=False, pin_memory=False, num_workers=0
)

for epoch in range(1):
    t0 = time.time()
    for i, data in enumerate(train_loader):
        print(data[0].shape, data[1].shape)
        break
