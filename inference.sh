python tools/inference.py cfgs/TeethSeg_models/AdaPoinTr.yaml /storage/share/repos/PoinTr/experiments/AdaPoinTr/TeethSeg_models/230908_TeethSeg_exp/ckpt-epoch-3000.pth --pc_root /storage/share/temp/3DTeethSeg22/data-processed/num_points-16384/inference/ --out_pc_root /storage/share/temp/3DTeethSeg22/data-processed/num_points-16384/inference-result/

python tools/inference.py cfgs/PCN_models/AdaPoinTr.yaml /storage/share/repos/PoinTr/experiments/AdaPoinTr/PCN_models/230908_PCN_exp/ckpt-last.pth --pc_root /storage/share/temp/PCN/inference/ --out_pc_root /storage/share/temp/PCN/inference-result/

