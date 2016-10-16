--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ImageNet and CIFAR-10 datasets
--

local M = {}

local function isvalid(opt, cachePath)
   local imageInfo = torch.load(cachePath)
   if imageInfo.basedir and imageInfo.basedir ~= opt.data then
      return false
   end
   return true
end

function M.create(opt, split)
  --folder where data is kept
  --checks if train/val data is in one file already
   local cachePath = paths.concat(opt.gen, opt.dataset .. '.t7')
   --
   -- download and create the dataset in .t7 if it does not exist
   if not paths.filep(cachePath) or not isvalid(opt, cachePath) then
      paths.mkdir('gen')

      -- dateset-gen.lua should exist. It does data setup
      local script = paths.dofile(opt.dataset .. '-gen.lua')
      script.exec(opt, cachePath)
   end

   -- imageInfo is a table with entries 
   -- for train and val data
   -- train and val are keys which point to 
   -- the actual images
   local imageInfo = torch.load(cachePath)

   local Dataset = require('datasets/' .. opt.dataset)
   return Dataset(imageInfo, opt, split)
end

return M
