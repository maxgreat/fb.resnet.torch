local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling

local function createModel(opt)

  local widthMult = opt.widthMult
  local baseWidth = opt.width
  local numClasses = 100

  local depthFilters = baseWidth*widthMult
  local model = nn.Sequential()

  model:add(Convolution(3, baseWidth,5,5,1,1,2,2))
  model:add(ReLU(true))

  model:add(Max(3,3,2,2):ceil())
  model:add(ReLU(true))
  
  model:add(nn.Dropout(0.5))

  model:add(Convolution(baseWidth, depthFilters,5,5,1,1,2,2))
  model:add(ReLU(true))

  model:add(Max(3,3,2,2):ceil())
  model:add(ReLU(true))

  model:add(nn.Dropout(0.5))

  model:add(Convolution(depthFilters,depthFilters,3,3,1,1,0,0))
  model:add(ReLU(true))

  model:add(Convolution(depthFilters,depthFilters,1,1,1,1,0,0))
  model:add(ReLU(true))

  model:add(Convolution(depthFilters,numClasses,1,1,1,1,0,0))
  model:add(ReLU(true))

  model:add(Avg(6, 6, 1, 1))
  model:add(nn.View(numClasses):setNumInputDims(3))
  model:add(nn.Linear(numClasses, numClasses))

  local function ConvInit(name)
    for k,v in pairs(model:findModules(name)) do
       local n = v.kW*v.kH*v.nOutputPlane
       v.weight:normal(0,math.sqrt(2/n))
       if cudnn.version >= 4000 then
          v.bias = nil
          v.gradBias = nil
       else
          v.bias:zero()
       end
    end
  end

  ConvInit('cudnn.SpatialConvolution')
  ConvInit('nn.SpatialConvolution')

  for k,v in pairs(model:findModules('nn.Linear')) do
    v.bias:zero()
  end

  model:cuda()

  if opt.cudnn == 'deterministic' then
    model:apply(function(m)
       if m.setMode then m:setMode(1,1,1) end
    end)
  end

  model:get(1).gradInput = nil

  return model
end

return createModel
