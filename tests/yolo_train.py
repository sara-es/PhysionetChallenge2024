import os, sys
sys.path.append(os.path.join(sys.path[0], '..'))
import yaml
from digitization import YOLOv7

args = YOLOv7.train.OptArgs()
args.epochs = 100
args.workers = 8
args.device = "0"
args.batch_size = 8
args.img_size = [640, 640]
args.data = os.path.join("digitization", "YOLOv7", "data", "ecg.yaml")
args.cfg = os.path.join("digitization", "YOLOv7", "cfg", "training", "yolov7-ecg2c.yaml")
args.weights = os.path.join("digitization", "model_checkpoints", "yolov7.pt")
args.name = "yolov7-ecg-2c-multi-res"
args.hyp = os.path.join("digitization", "YOLOv7", "data", "hyp.scratch.custom.yaml")
args.cache_images = False
args.multi_scale = True

if __name__ == "__main__":
    import torch
    print(f"Cuda available: {torch.cuda.is_available()}.")
    YOLOv7.train.main(args)

