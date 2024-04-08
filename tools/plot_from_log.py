import os
import re
import numpy as np
import argparse
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('log_file', type=str)
parser.add_argument('eval_interval', type=int, help='e.g., 1000')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()
if args.debug:
    import pdb; pdb.set_trace()

results_dict = OrderedDict()


with open(args.log_file, 'r') as f:
    iter_num = -1
    for line in f.readlines():
        line = line.strip()
        if line == '':
            continue
        
        #if 'Saving checkpoint to' in line:
        #    iter_num = line.split('/')[-1].split('_')[-1].split('.')[0]
        #    if iter_num not in results_dict:
        #        results_dict[iter_num] = []

        if ('d2.utils.events INFO:  eta' in line) and ('iter:' in line):
            v = re.findall(r'iter: [\d]+', line)
            assert len(v) == 1
            iter_num = float(v[0].strip().split(' ')[-1])
            iter_num = round(iter_num / args.eval_interval) * args.eval_interval

        if ('d2.evaluation.testing INFO: copypaste' in line) and ('Task' not in line) and ('AP' not in line):
            if iter_num not in results_dict:
                results_dict[iter_num] = []
            
            ap, ap50 = line.split(':')[-1].strip().split(',')[:2]
            if len(results_dict[iter_num]) >= 2:
                assert (ap, ap50) in results_dict[iter_num]
            else:
                results_dict[iter_num].append((ap, ap50))

ap_student = []
ap_teacher = []
ap50_student = []
ap50_teacher = []

iters = list(results_dict.keys())

for i in iters:
    assert len(results_dict[i]) == 2, 'key: {}, value: {}'.format(i, results_dict[i])
    ap_s, ap50_s = results_dict[i][0]
    ap_t, ap50_t = results_dict[i][1]

    ap_student.append(ap_s)
    ap_teacher.append(ap_t)
    ap50_student.append(ap50_s)
    ap50_teacher.append(ap50_t)

ap50_teacher = [float(i) for i in ap50_teacher] 
ap_teacher = [float(i) for i in ap_teacher]  

print('*' * 40)
print('iter_num')
print('  {}\n'.format(iters))

print('ap_teacher')
print('  {}'.format(ap_teacher))

idx = np.argmax(ap_teacher)
print('max iter, ap and ap50: {} | {} | {}'.format(iters[idx], ap_teacher[idx], ap50_teacher[idx]))