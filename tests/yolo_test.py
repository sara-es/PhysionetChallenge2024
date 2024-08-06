import os, sys
sys.path.append(os.path.join(sys.path[0], '..'))

from digitization import YOLOv7
from digitization.YOLOv7.utils.general import check_file

if __name__ == "__main__":
    args = YOLOv7.detect.OptArgs()
    args.weights = os.path.join("runs","train","yolov7-ecg2c-multi-res","weights","best.pt")
    args.save_txt = True
    args.save_conf = True
    args.device = "0"
    args.source = os.path.join("tiny_testset", "real_images", "images")
    YOLOv7.detect.main(args)
    