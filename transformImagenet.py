import numpy as np
import lmdb
import glob
import sys
import os
from PIL import Image

size = 128

if len(sys.argv) < 3:
	print('Usage : python createLMDB imagenet <path to imagenet> <path to save dir>')
	sys.exit(1)

trainpath = os.path.join(sys.argv[1], 'train/*')
valpath = os.path.join(sys.argv[1], 'val/*')

if not os.path.isdir(trainpath):
	os.mkdir(trainpath)
for concept in glob.glob(trainpath):
	print('Concept :', concept)
	for imName in glob.glob(os.path.join(concept, '*')):
		#print("Image :", imName)
		im = Image.open(imName).convert("RGB")
		im = im.resize((size,size))
		
		savedDir = os.path.join(sys.argv[2], "train", os.path.basename(concept))
		if not os.path.isdir(savedDir):
			os.mkdir(savedDir)
		im.save(os.path.join(savedDir, os.path.basename(imName)))

if not os.path.isdir(valpath):
	os.mkdir(valpath)
for concept in glob.glob(valpath):
	for imName in glob.glob(os.path.join(concept, '*')):
		im = Image.open(imName).convert("RGB")
		im = im.resize((size,size))

		savedDir = os.path.join(sys.argv[2], "val", os.path.basename(concept))

		if not os.path.isdir(savedDir):
			os.mkdir(savedDir)
		im.save(os.path.join(savedDir, os.path.basename(imName)))
