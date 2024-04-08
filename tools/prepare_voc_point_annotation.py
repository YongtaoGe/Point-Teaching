import os 
import argparse
from lxml import etree, objectify
import numpy as np
import glob
from tqdm import tqdm

def instance2xml_base(filename, width, height, depth, segmented):
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('pascal voc'),
        E.filename(filename),
        E.source(
            E.database('pascal voc'),
            E.annotation('pascal voc'),
            E.image(''),
            E.url('')
        ),
        E.size(
            E.width(width),
            E.height(height),
            E.depth(depth)
        ),
        E.segmented(segmented),
    )
    return anno_tree


def worker(xml_file, out_dir):
    root = etree.parse(xml_file)
    height = int(float(root.find('size').find('height').text))
    width = int(float(root.find('size').find('width').text))
    depth = int(float(root.find('size').find('depth').text))
    segmented = root.find('segmented').text
    ann_tree = instance2xml_base(os.path.splitext(os.path.basename(xml_file))[0], width, height, depth, segmented)

    for obj in root.findall('object'):
        cls_name = obj.find('name').text

        xmin = obj.find('bndbox').find('xmin').text
        ymin = obj.find('bndbox').find('ymin').text
        xmax = obj.find('bndbox').find('xmax').text
        ymax = obj.find('bndbox').find('ymax').text

        difficult = obj.find('difficult').text

        # generate point
        try:
            p_x = np.random.randint(int(float(xmin)) + 1, int(float(xmax)) - 1)
            p_y = np.random.randint(int(float(ymin)) + 1, int(float(ymax)) - 1)
        except:
            p_x = np.random.randint(int(float(xmin)), int(float(xmax)))
            p_y = np.random.randint(int(float(ymin)), int(float(ymax)))

        E = objectify.ElementMaker(annotate=False)
        anno = E.object(
            E.name(cls_name),
            E.bndbox(
                E.xmin(xmin),
                E.ymin(ymin),
                E.xmax(xmax),
                E.ymax(ymax)
            ),
            E.point(
                E.x(p_x),
                E.y(p_y)
            ),
            E.difficult(difficult)
        )
        ann_tree.append(anno)

    out_xml_file = os.path.join(out_dir, os.path.basename(xml_file))
    etree.ElementTree(ann_tree).write(out_xml_file, pretty_print=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('xml_dir', type=str)
    parser.add_argument('out_xml_dir', type=str)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug == True:
        import pdb; pdb.set_trace()

    xml_files = list(glob.glob(os.path.join(args.xml_dir, '*.xml')))

    os.makedirs(args.out_xml_dir)

    for i in tqdm(range(len(xml_files))):
        xml_file = xml_files[i]
        worker(xml_file, args.out_xml_dir)


# python prepare_voc_point_annotation.py --xml_dir ./datasets/VOC2007/Annotations --out_xml_dir ./datasets/VOC2007/Annotations_w_points