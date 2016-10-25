require 'paths'

local cmd = torch.CmdLine()

cmd:text()
cmd:text('Create the list of image with the correspondant class')
cmd:text()
cmd:text('Options:')
------------ General options --------------------
cmd:option('-data',       '',         'Path to the image list file')
cmd:option('-output',    'stdout', 'output image list file')
cmd:option('-nImages',        100, 'number of image to take')
cmd:text()

local opt = cmd:parse(arg or {})

imList = torch.load(opt.data)

if opt.output == 'stdout' then
	for i=1,#imList.image_list do
		if i <= opt.nImages then
			a, n = torch.max(imList.features[i], 1)
			print(imList.image_list[i].." "..n[1])
		end
	end
end