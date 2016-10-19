require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'
require 'paths'
local t = require '../datasets/transforms'

if #arg < 4 then
   io.stderr:write('Usage : th extract-imagenet.lua [MODEL] [BATCHSIZE] [IMAGENETDIR] [OUPUTDIR]\n')
   os.exit(1)
end



function extract_features (params)
  --local f = assert(loadfile("extract-features.lua"))(params)
  --return f()
  --print("call ".."lua extract-features.lua "..params[1].." "..params[2].." "..params[3].." "..params[4])
  os.execute("th extract-features.lua "..params[1].." "..params[2].." "..params[3].." "..params[4])
end

batch_size = tonumber(arg[2])
dir_path   = arg[3]
outputdir  = arg[4]
for file in paths.iterdirs(dir_path) do --for each classes
	if not paths.filep(outputdir..'/'..file..'.t7') then
		print("Extract features for concept : "..file..'.t7')
		local params = {}
		params[1] = arg[1] --model
		params[2] = batch_size
		params[3] = outputdir..'/'..file..'.t7' --file to save features
		params[4] = dir_path..file --directory with images
		extract_features(params) --call extract-features.lua
	end
end