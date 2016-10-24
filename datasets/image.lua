local t = require 'datasets/transforms'

local M = {}
local ImageDataset = torch.class('resnet.ImageDataset', M)

function ImageDataset:__init(imageInfo, opt, split)
  assert(imageInfo[split], split)

  self.imageInfo = imageInfo[split]
  self.split = split
end

function ImageDataset:get(i)
  local image = self.imageInfo.data[i]:float()
  local label = self.imageInfo.labels[i]
  return {
    input = image,
    target = label,
  }
end

function ImageDataset:size()
  -- print (#self.imageInfo.data)
  --return #self.imageInfo.data
  return self.imageInfo.data:size(1)
end

---- Preprocess options
--- values for CIFAR
-- make generic, by computing from loaded data
local meanstd = {
  mean = {125.3, 123.0, 113.9},
  std =  {63.0, 62.1, 66.7},
}
function ImageDataset:preprocess()
  if self.split == 'train' then
    return t.Compose{
      t.ColorNormalize(meanstd),
      t.HorizontalFlip(0.5),
      t.RandomCrop(32, 4),
    }
  elseif self.split == 'val' then
    return t.ColorNormalize(meanstd)
  else
    error('Invalid split:' .. self.split)
  end
end

return M.ImageDataset
