require 'paths'

local function setToMaxOne(vec)
	a, i = torch.max(vec, 1)
	return torch.zeros(vec:size()):scatter(1,i,1)
end

local function numberOfOne(vec)
	local n = 0
	for i=1,vec:size()[1] do
		if vec[i] == 1 then
			n = n+1
		end
	end
	return n
end

local function LoadAndFormat(filename)
	res = torch.load(filename)
	for i=1,res.features:size()[1] do
		res.features[i] = setToMaxOne(res.features[i])
	end
	return res
end

local cmd = torch.CmdLine()

cmd:text()
cmd:text('Set data to training structure')
cmd:text()
cmd:text('Options:')
------------ General options --------------------
cmd:option('-data',       '',         'Path to the .t7 file(s)')
cmd:option('-output',    'onehot', 'Options: onehot | ... (only onehot for now)')
cmd:option('-outputDir', './gen/', 'Path to the output directory')
cmd:option('-nThreads',        1, 'number of data loading threads (only one for now)')
cmd:text()

local opt = cmd:parse(arg or {})

if paths.extname(opt.data) == 't7' then
	--Only one file to do
	if not paths.filep(opt.outputDir..paths.basename(opt.data)) then
		r = LoadAndFormat(file)
		torch.save(opt.outputDir..paths.basename(opt.data), r)
	else
		print("File "..paths.basename(opt.data).."Already exist in "..opt.outputDir)
	end
elseif paths.dirp(opt.data) then
	for file in paths.iterfiles(opt.data) do --for each .t7 file
		if paths.extname(file) == 't7' and not paths.filep(opt.outputDir..file) then
			print("Handling file : "..file) 
			r = LoadAndFormat(opt.data..file)
			torch.save(opt.outputDir..paths.basename(file), r)
		end
	end
else
	print("Unknow file "..opt.data)
end
