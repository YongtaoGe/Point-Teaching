DATASET=coco_2017_train_points_n2_v1
EXP_NAME=r50_coco_10.0
OUTPUT_DIR=results/semi_weak_sup/frcnn/${EXP_NAME}
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

OMP_NUM_THREADS=1 python train_net.py \
       --resume \
       --num-gpus 8 \
       --config-file configs/Faster-RCNN/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1_mil.yaml \
       --dist-url tcp://127.0.0.1:50159 \
       OUTPUT_DIR ${OUTPUT_DIR} \
       DATASETS.TRAIN "('${DATASET}',)" \
       SOLVER.IMS_PER_BATCH_LABEL 32 \
       SOLVER.IMS_PER_BATCH_UNLABEL 32 \
       DATALOADER.SUP_PERCENT 10.0 \
       DATALOADER.NUM_WORKERS 4 \
       TEST.EVAL_PERIOD 2000 \
       SOLVER.CHECKPOINT_PERIOD 2000 \
       SEMISUPNET.Trainer "faster_rcnn_point_sup" \
       SEMISUPNET.PSEUDO_BBOX_SAMPLE "hungarian" \
       SEMISUPNET.BURN_UP_STEP 2000 \
       SEMISUPNET.POINT_LOSS_WEIGHT 0.0 \
       SEMISUPNET.IMG_MIL_LOSS_WEIGHT 1.0 \
       SEMISUPNET.INS_MIL_LOSS_WEIGHT 0.1 \
       SOLVER.STEPS "(179990, 179995)" \
       SOLVER.MAX_ITER 180000 \
       SEMISUPNET.BBOX_THRESHOLD 0.05 \
       TEST.VAL_LOSS False \
       DATALOADER.RANDOM_DATA_SEED_PATH "datasets/coco/dataseed/COCO_supervision_117266.txt" \
       SOLVER.LR_SCHEDULER_NAME "WarmupTwoStageMultiStepLR" \
       SOLVER.FACTOR_LIST "(1, 0.1, 0.25, 0.5, 0.75, 1, 0.1, 0.01)" \
       SOLVER.STEPS "(2000, 3000, 4000, 5000, 6000, 179990, 179995)" \
       # DATALOADER.RANDOM_DATA_SEED_PATH "./dataseed/COCO_supervision_117266_blance_sampled.txt" \

