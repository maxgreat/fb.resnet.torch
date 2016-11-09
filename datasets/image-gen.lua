local sys = require 'sys'
local image = require 'image'
local paths = require 'paths'

local M = {}


local function loadAndConvertImages(imageList, baseDir, numExamples)
  local data = torch.Tensor(numExamples,3,256,256):float()
  local labels = torch.Tensor(numExamples):float()
  local image_dimensions
 
  idx = 0
  local f = io.open(imageList)
  assert(f ~= nil, ' File does not exist: ' .. imageList)
  while idx < numExamples do
    print(idx)
    idx = idx + 1
    local line = f:read('*line')
    if not line then break end

    -- split line into filename and label
    local image_data = {}
    for word in line:gmatch("%S+") do
      table.insert(image_data, word)
    end

    local image_file = paths.concat(baseDir, image_data[1])
    -- add 1 to label to have non-zero labels for each example
    local label = tonumber(image_data[2]) -- + 1

    local ok, input_image = pcall(function()
      return image.load(image_file, 3, 'byte')
    end)

    input_image = image.scale(input_image,256,256)
    if ok then 
        data[idx] = torch.Tensor(3, 256, 256):copy(input_image)
        labels[idx] = label
    else
      print('Cannot load: ' .. image_file)
    end
  end
  f:close()

  return {
    data = data,
    labels = torch.Tensor(labels):float()
  }
end

function M.exec(opt, cacheFile)
  -- file containing path to image and associated label
  local trainList = opt.trainList
  local valList = opt.valList
  print("Train list creation")
  local trainData = loadAndConvertImages(trainList, opt.trainBaseDir, 50000) 
  print("Val list creation")
  local testData = loadAndConvertImages(valList, opt.valBaseDir, 50000)

  print("-- Saving dataset to " .. cacheFile)
  torch.save(cacheFile, {
    train = trainData,
    val = testData,
  })


end

return M
