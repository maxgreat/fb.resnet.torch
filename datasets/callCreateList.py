import glob
import subprocess

for image in glob.glob('../pretrained/res/*.t7'):
	#print "Call th createListImage.lua -data ", image, ' -nImages 5 >> imageList.txt'
	subprocess.call(['th', 'createListImage.lua', '-data', image, '-nImages', '100'])
