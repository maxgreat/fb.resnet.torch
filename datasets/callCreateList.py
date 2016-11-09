import glob
import subprocess

nbImages = '500'

with open("trainList.txt", "w") as ftrain:
	for image in glob.glob('../pretrained/res/*.t7'):
		subprocess.call(['th', 'createListImage.lua', '-data', image, '-nImages', nbImages], stdout=ftrain)

with open("valList.txt", "w") as f:
	for image in glob.glob('../pretrained/val/*.t7'):
		subprocess.call(['th', 'createListImage.lua', '-data', image, '-nImages', nbImages],stdout=f)
