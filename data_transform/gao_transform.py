import sys
import os.path
import shutil
import argparse
import xml.etree.ElementTree as ET


def change_content(xml_path: str, root_name: str, context: str):
	for xml_file in file_gen(xml_path, ".xml"):
		tree = ET.parse(xml_file)
		root = tree.getroot()
		shutil.copy(xml_file, xml_file+'.old')
		for name in root.iter(root_name):
			name.text = context
		tree.write(xml_file)

def parse_gao(xml_file):
	root = ET.parse(xml_file).getroot()
	for obj in root.iter('object'):
		obj_struct = {}
		obj_struct['name'] = next(x for x in obj.iter('name')).text
		bbox = []
		for points in obj.iter('points'):
			for p in list(points.iter('point'))[:-1]:
				bbox += list(map(float, p.text.split(',')))
		obj_struct["bbox"] = bbox
		yield obj_struct

def file_gen(root_dir, ext = None):
	for root, _, files in os.walk(root_dir):
		for file in files:
			if (ext and file.endswith(ext)) or ext is None:
				yield os.path.join(root, file)

def basename(path):
	return os.path.splitext(os.path.basename(path))[0]

def dots4ToRec4(poly):
	xmin = min(poly[0], poly[2], poly[4], poly[6])
	xmax = max(poly[0], poly[2], poly[4], poly[6])
	ymin = min(poly[1], poly[3], poly[5], poly[7])
	ymax = max(poly[1], poly[3], poly[5], poly[7])
	return xmin, ymin, xmax, ymax

def bbox2darknet(poly, img_w, img_h):
	xmin, ymin, xmax, ymax = dots4ToRec4(poly)
	x = (xmin + xmax)/2
	y = (ymin + ymax)/2
	w = xmax - xmin
	h = ymax - ymin
	return x/img_w, y/img_h, w/img_w, h/img_h

def gao2darknet(xml_path: str, txt_path: str, img_w: int, img_h: int):
	names = {}
	os.makedirs(txt_path, exist_ok=True)
	for xml_file in file_gen(xml_path, ".xml"):
		txt_file = os.path.join(txt_path, basename(xml_file) + '.txt')
		with open(txt_file, 'w') as f_out:
			objects = parse_gao(xml_file)
			for obj in objects:
				name = obj['name']
				if name not in names:
					names[name] = len(names)
				bbox = bbox2darknet(obj['bbox'], img_w, img_h)
				outline = str(names[name]) + ' ' + ' '.join(list(map(str, bbox)))
				f_out.write(outline + '\n')
	return names

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Converts Gaofen Dataset to Yolo Format')
	parser.add_argument('--xml', metavar='labels_xml/', type=str,
				help='Gaofen_XML Path', required=True)
	parser.add_argument('--txt', metavar='labels_txt/', type=str,
				help='Yolo_TXT Path to Create', required=True)
	parser.add_argument('--img_w', metavar='1024', type=int,
				help='Image width', required=True)
	parser.add_argument('--img_h', metavar='1024', type=int,
				help='Image height', required=True)
	parser.add_argument('--change', metavar='airplane', type=str,
				help='Change all class names to the given')
	if len(sys.argv) < 4:
		parser.print_help(sys.stderr)
		sys.exit(1)
	args = parser.parse_args()
	if args.change:
		change_content(args.xml, 'name', args.change)
	names = gao2darknet(args.xml, args.txt, args.img_w, args.img_h)
	print(names)