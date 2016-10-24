local sys = require 'sys'
local image = require 'image'
local paths = require 'paths'

local M = {}

--local function loadAndConvertImages(imageList, baseDir)
--  local data = {}
--  local labels = {}
--
--  local f = io.open(imageList)
--  assert(f ~= nil , 'File does not exist: ' .. imageList)
--  while true do
--    local line = f:read('*line')
--    if not line then break end
--
--    local image_data = {}
--    for word in line:gmatch("%S+") do
--      table.insert(image_data, word)
--    end
--
--    local image_file = paths.concat(baseDir, image_data[1])
--    local label = tonumber(image_data[2]) + 1
--
--    local ok, input_image = pcall(function()
--      return image.load(image_file, 3, 'float')
--    end)
--
--    if ok then
--      table.insert(data, input_image)
--      table.insert(labels, label)
--    else
--      print('Cannot load: ' .. image_file)
--    end
--  end
--  f:close()
--
--  return {
--    data = data, 
--    labels = labels,
--  }
--
--
--end

local function loadAndConvertImages(imageList, baseDir)
  local data, image_dimensions
  -- using labels as a table
  -- need to figure out similar method for converting
  -- the table of images to tensor
  local labels = {}

  idx = 0
  local f = io.open(imageList)
  assert(f ~= nil, ' File does not exist: ' .. imageList)
  while true do
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
    local label = tonumber(image_data[2]) + 1

    local ok, input_image = pcall(function()
      return image.load(image_file, 3, 'byte')
    end)

    if ok then 
      if data == nil then
        print 'setting up the tensors to hold all the images'
      
        image_dimensions = input_image:size()
        data = torch.Tensor(1, image_dimensions[1], image_dimensions[2], image_dimensions[3]):byte()
        data[1] = input_image 
        --labels = torch.Tensor(1):float()
        --labels[1] = label
        table.insert(labels, label)
      else
        -- reshape image from 3x32x32 to 1x3x32x32
        -- to concat into the big tensor for training images
        -- possible better solution
        data = torch.cat(data,
               input_image:reshape(1, image_dimensions[1], image_dimensions[2], image_dimensions[3]),
               1)
        table.insert(labels, label)
      end
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

  local trainData = loadAndConvertImages(trainList, opt.trainBaseDir)
  local testData = loadAndConvertImages(valList, opt.valBaseDir)

  print("-- Saving dataset to " .. cacheFile)
  torch.save(cacheFile, {
    train = trainData,
    val = testData,
  })


end

return M
