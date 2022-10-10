pip install pycocotools==2.0.5 antlr4-python3-runtime==4.9.3

ln -s /home2/pytorch-broad-models/COCO2017 datasets/coco

python tools/train_net.py --config-file configs/Misc/cascade_mask_rcnn_R_50_FPN_1x.yaml --eval-only MODEL.WEIGHTS /home2/pytorch-bbrad-models/Cascade_R-CNN_ResNet101-FPN/Cascade_R-CNN_ResNet50-FPN.pkl MODEL.DEVICE cpu