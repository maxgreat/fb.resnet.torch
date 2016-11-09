--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
opt.save = opt.dataset .. 'checkpoints'
print(opt)

torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)
-- save checkpoint in a folder with same name as dataset

-- Create model
local model, criterion = models.setup(opt, checkpoint)
print(model)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)


if opt.testOnly then
  local top1Err, top5Err = trainer:test(0, valLoader)
  print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
  return
end

-- Logger settings
local logger = optim.Logger(opt.save, 'test.log')
logger:display(false)
logger.showPlot = false
logger:setNames{'Train acc', 'Test acc.'}
logger:style({'+-', '+-'})

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestTop1 = math.huge
local bestTop5 = math.huge
for epoch = startEpoch, opt.nEpochs do

  local timer = torch.Timer()
  -- Train for a single epoch
  local trainTop1, trainTop5, trainLoss = trainer:train(epoch, trainLoader)

  -- Run model on validation set
  local testTop1, testTop5 = trainer:test(epoch, valLoader)

  logger:add{trainTop1, testTop1}
  local bestModel = false
  if testTop1 < bestTop1 then
    bestModel = true
    bestTop1 = testTop1
    bestTop5 = testTop5

  end
  checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
  logger:plot()
end

print(string.format(' * Finished top1: %6.3f  top5: %6.3f', bestTop1, bestTop5))
