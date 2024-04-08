DATASET=coco_2017_train_points_n2_v2
#PROPOSAL_FILES_TRAIN="detectron2://COCO-Detection/rpn_R_50_FPN_1x/137258492/coco_2017_train_box_proposals_21bc3a.pkl"
#PROPOSAL_FILES_TRAIN="./datasets/coco/coco_2017_train_box_proposals_21bc3a.pkl"
EXP_NAME=r50_coco_1_cp_finetune
OUTPUT_DIR=results/semi_weak_sup/mrcnn/${EXP_NAME}
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

OMP_NUM_THREADS=1 python train_net.py \
       --num-gpus 8 \
       --config-file configs/Mask-RCNN/coco_supervision/mask_rcnn_R_50_FPN_sup1_run1.yaml \
       --dist-url tcp://127.0.0.1:50359 \
       OUTPUT_DIR ${OUTPUT_DIR} \
       DATASETS.TRAIN "('${DATASET}',)" \
       SOLVER.IMS_PER_BATCH_LABEL 16 \
       SOLVER.IMS_PER_BATCH_UNLABEL 16 \
       SOLVER.CHECKPOINT_PERIOD 5000 \
       DATALOADER.SUP_PERCENT 1.0 \
       DATALOADER.NUM_WORKERS 4 \
       TEST.EVAL_PERIOD 10 \
       SEMISUPNET.Trainer "mask_rcnn_baseline" \
       SOLVER.MAX_ITER 60000 \
       SOLVER.STEPS "(44999, 54999)" \
       TEST.VAL_LOSS False \
       DATALOADER.RANDOM_DATA_SEED_PATH "datasets/coco/dataseed/COCO_supervision_117266.txt" \

      #  MODEL.WEIGHTS "${OUTPUT_DIR}/model_final.pth" \
      #  SEMISUPNET.PSEUDO_BBOX_SAMPLE "hungarian" \
      #  SEMISUPNET.POINT_LOSS_WEIGHT 0.0 \
      #  SEMISUPNET.IMG_MIL_LOSS_WEIGHT 1.0 \
      #  SEMISUPNET.INS_MIL_LOSS_WEIGHT 0.1 \
    #    MODEL.WEIGHTS "./results/semi_weak_sup/frcnn/r50_coco_10.0_cp_v4/model_0139999.pth" \
    #    SOLVER.LR_SCHEDULER_NAME "WarmupTwoStageMultiStepLR" \
    #    SOLVER.STEPS "(2500, 3000, 4000, 5000, 6000, 89990, 89995)" \
    #    SOLVER.FACTOR_LIST "(1, 0.1, 0.25, 0.5, 0.75, 1, 0.1, 0.01)" \