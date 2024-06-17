from tidecv import TIDE, datasets

tide = TIDE()
ground_truth = 'datasets/coco/annotations/instances_val2017.json'
# json_results = 'results/semi_weak_sup/frcnn/r50_coco_30.0_cp_v4/inference/coco_instances_results.json'
# json_results = 'results/semi_weak_sup/frcnn/r50_coco_10.0_cp_v4/inference/coco_instances_results.json'
# json_results = 'results/pteacher/inference/coco_instances_results.json'
# json_results = 'results/semi_weak_sup/mrcnn/r50_coco_10_point_match_ins_mil/inference/coco_instances_results.json'

json_results = 'results/point_teaching_pseudo/inference/coco_instances_results.json'
# json_results = 'results/pteacher/inference/coco_instances_results.json'
tide.evaluate(datasets.COCO(path=ground_truth), datasets.COCOResult(json_results), mode=TIDE.BOX) # Use TIDE.MASK for masks
# tide.evaluate(datasets.COCO(path=ground_truth), datasets.COCOResult(json_results), mode=TIDE.MASK) # Use TIDE.MASK for masks
tide.summarize()  # Summarize the results as tables in the console
tide.plot('./')   # Show a summary figure. Specify a folder and it'll output a png to that folder.
