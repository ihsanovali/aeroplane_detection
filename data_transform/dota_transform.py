import sys
import os.path
import argparse
from PIL import Image

def parse_dota(filename):
	with open(filename,"r+") as f:
		for line in f:
			splitlines = line.strip().split(' ')
			obj_struct = {}
			if len(splitlines) < 9:
				continue
			if len(splitlines) >= 9:
				obj_struct['name'] = splitlines[8]
			obj_struct['bbox'] = list(map(float, splitlines[0:8]))
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

def dota2darknet(txt_path: str, img_path: str, yolo_path: str, names: dict):
	filter = len(names) != 0
	os.makedirs(yolo_path, exist_ok=True)
	for file in file_gen(txt_path, ".txt"):
		txt_file = os.path.join(yolo_path, basename(file) + '.txt')
		img_file = os.path.join(img_path, basename(file) + '.png')
		if os.path.isfile(img_file):
			img = Image.open(img_file)
			with open(txt_file, 'w') as f_out:		
				objects = parse_dota(file)
				for obj in objects:
					name = obj['name']
					if not filter and name not in names:
						names[name] = len(names)
					if name in names: 
						img_w, img_h = img.size
						bbox = bbox2darknet(obj['bbox'], img_w, img_h)
						outline = str(names[name]) + ' ' + ' '.join(list(map(str, bbox)))
						f_out.write(outline + '\n')
	return names

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Converts Dota Dataset to Yolo Format')
	parser.add_argument('--dota_txt', metavar='labels/', type=str,
				help='Dota_TXT Path', required=True)
	parser.add_argument('--dota_img', metavar='images/', type=str,
				help='Dota_IMG Path', required=True)
	parser.add_argument('--yolo', metavar='labels_yolo/', type=str,
				help='Yolo_TXT Path to Create', required=True)

	if len(sys.argv) < 4:
		parser.print_help(sys.stderr)
		sys.exit(1)
	args = parser.parse_args()
	names = dota2darknet(args.dota_txt, args.dota_img, args.yolo, names = {'plane':0})
	print(names)