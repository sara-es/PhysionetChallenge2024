import os, sys
sys.path.append(os.path.join(sys.path[0], '..'))

from digitization import YOLOv7
from digitization.YOLOv7.utils.general import check_file

if __name__ == "__main__":
    args = YOLOv7.test.OptArgs()
    args.data = os.path.join("digitization", "YOLOv7", "data", "ecg.yaml")
    args.weights = os.path.join("runs","train","yolov7-ecg2c-multi-res","weights","best.pt")
    args.batch_size = 8
    args.save_txt = True
    args.save_conf = True
    args.device = "0"
    YOLOv7.test.main(args)
    