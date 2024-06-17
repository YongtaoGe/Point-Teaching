
<div align="center">
<h1>
Point-teaching: Weakly semi-supervised Object Detection with Point Annotations (AAAI 2023)
</h1>
</div>
<br/>

## Install
```
pip install detectron2
pip install -e . -v
```

## Train
```
# 10% coco subset
sh scripts/train_frcnn_10.0_ins_mil_point_match.sh
```

## Test
```
sh scripts/test_frcnn.sh
```

## 🎓 Citation
```
@inproceedings{ge2023point,
  title={Point-teaching: Weakly semi-supervised object detection with point annotations},
  author={Ge, Yongtao and Zhou, Qiang and Wang, Xinlong and Shen, Chunhua and Wang, Zhibin and Li, Hao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={1},
  pages={667--675},
  year={2023}
}
```