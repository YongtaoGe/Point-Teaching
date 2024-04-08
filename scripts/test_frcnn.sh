DATASET=coco_2017_val_points_n1_v0
EXP_NAME=point_teaching_pseudo
OUTPUT_DIR=results/${EXP_NAME}
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

OMP_NUM_THREADS=1 python train_net.py \
       --eval-only \
       --num-gpus 1 \
       --config-file configs/Faster-RCNN/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1_mil.yaml \
       --dist-url tcp://127.0.0.1:50359 \
       OUTPUT_DIR ${OUTPUT_DIR} \
       DATASETS.TRAIN "('${DATASET}',)" \
       DATASETS.TEST "('${DATASET}',)" \
       SOLVER.IMS_PER_BATCH_LABEL 8 \
       SOLVER.IMS_PER_BATCH_UNLABEL 8 \
       DATALOADER.SUP_PERCENT 1.0 \
       DATALOADER.NUM_WORKERS 2 \
       TEST.EVAL_PERIOD 5000 \
       SOLVER.CHECKPOINT_PERIOD 5000 \
       SEMISUPNET.Trainer "faster_rcnn_point_sup" \
       SEMISUPNET.PSEUDO_BBOX_SAMPLE "hungarian" \
       SEMISUPNET.BURN_UP_STEP 36000 \
       SEMISUPNET.POINT_LOSS_WEIGHT 0.0 \
       SEMISUPNET.IMG_MIL_LOSS_WEIGHT 1.0 \
       SEMISUPNET.INS_MIL_LOSS_WEIGHT 0.1 \
       SEMISUPNET.BBOX_THRESHOLD 0.05 \
       SOLVER.MAX_ITER 36000 \
       TEST.VAL_LOSS False \
       DATALOADER.RANDOM_DATA_SEED_PATH "datasets/coco/dataseed/COCO_supervision_117266.txt" \
       SOLVER.LR_SCHEDULER_NAME "WarmupTwoStageMultiStepLR" \
       SOLVER.FACTOR_LIST "(1, 0.1, 0.01)" \
       SOLVER.STEPS "(24000, 32000)" \
       MODEL.WEIGHTS "./results/semi_weak_sup/frcnn/r50_coco_10.0_cp_v4_resume_jianchong/model_0011999.pth"



