--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Options:')
  ------------ General options --------------------
  cmd:option('-data',       '',         'Path to dataset')
  cmd:option('-dataset',    'cifar10', 'Options: imagenet | cifar10 | cifar100 | image')
  cmd:option('-dataFile',   '',        'Name of .t7 file to use for training and testing. This is needed in when using image.lua')
  cmd:option('-manualSeed', 0,          'Manually set RNG seed')
  cmd:option('-nGPU',       1,          'Number of GPUs to use by default')
  cmd:option('-backend',    'cudnn',    'Options: cudnn | cunn')
  cmd:option('-cudnn',      'fastest',  'Options: fastest | default | deterministic')
  cmd:option('-gen',        'gen',      'Path to save generated files')
  cmd:option('-debug',        'true',      'Debug mode')
  ------------- Data options ------------------------
  cmd:option('-nThreads',        2, 'number of data loading threads')
  cmd:option('-trainList',   '', 'A file which has the path to train images')
  cmd:option('-valList',     '', 'A file which has the path to validation images')
  cmd:option('-trainBaseDir',     '', 'Base Directory of train images')
  cmd:option('-valBaseDir',     '', 'Base Directory of val images')
  ------------- Training options --------------------
  cmd:option('-nEpochs',         0,       'Number of total epochs to run')
  cmd:option('-epochNumber',     1,       'Manual epoch number (useful on restarts)')
  cmd:option('-batchSize',       128,      'mini-batch size (1 = pure stochastic)')
  cmd:option('-testOnly',        'false', 'Run on validation set only')
  cmd:option('-tenCrop',         'false', 'Ten-crop testing')
  ------------- Checkpointing options ---------------
  cmd:option('-save',            'checkpoints', 'Directory in which to save checkpoints')
  cmd:option('-resume',          'none',        'Resume from the latest checkpoint in this directory')
  ---------- Optimization options ----------------------
  cmd:option('-LR',              0.01,   'initial learning rate')
  cmd:option('-momentum',        0.9,   'momentum')
  cmd:option('-weightDecay',     1e-4,  'weight decay')
  ---------- Model options ----------------------------------
  cmd:option('-netType',      'all_cnn', 'Options: resnet | preresnet | all_cnn')
  cmd:option('-depth',        34,       'ResNet depth: 18 | 34 | 50 | 101 | ...', 'number')
  cmd:option('-shortcutType', '',       'Options: A | B | C')
  cmd:option('-retrain',      'none',   'Path to model to retrain with')
  cmd:option('-optimState',   'none',   'Path to an optimState to reload from')
  ---------- Model options ----------------------------------
  cmd:option('-shareGradInput',  'false', 'Share gradInput tensors to reduce memory usage')
  cmd:option('-optnet',          'false', 'Use optnet to reduce memory usage')
  cmd:option('-resetClassifier', 'false', 'Reset the fully connected layer for fine-tuning')
  cmd:option('-nClasses',         0,      'Number of classes in the dataset')
  ---------- All CNN options ----------------------------------
  cmd:option('-width', 96, 'Number of filters in the first conv layer')
  cmd:option('-widthMult', 2, 'Multiplcation factor for conv filteres in deeper layers')
  cmd:text()

  local opt = cmd:parse(arg or {})

  opt.testOnly = opt.testOnly ~= 'false'
  opt.tenCrop = opt.tenCrop ~= 'false'
  opt.shareGradInput = opt.shareGradInput ~= 'false'
  opt.optnet = opt.optnet ~= 'false'
  opt.resetClassifier = opt.resetClassifier ~= 'false'

  if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
    cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
  end

  --SET DATASET
  --- IMAGENET
  if opt.dataset == 'imagenet' then
    -- Handle the most common case of missing -data flag
    local trainDir = paths.concat(opt.data, 'train')
    if not paths.dirp(opt.data) then
      cmd:error('error: missing ImageNet data directory')
    elseif not paths.dirp(trainDir) then
      cmd:error('error: ImageNet missing `train` directory: ' .. trainDir)
    end
    -- Default shortcutType=B and nEpochs=90
    opt.shortcutType = opt.shortcutType == '' and 'B' or opt.shortcutType
    opt.nEpochs = opt.nEpochs == 0 and 90 or opt.nEpochs
    --
    --- CIFAR10
  elseif opt.dataset == 'cifar10' then
    -- Default shortcutType=A and nEpochs=164
    opt.shortcutType = opt.shortcutType == '' and 'A' or opt.shortcutType
    opt.nEpochs = opt.nEpochs == 0 and 164 or opt.nEpochs
    --
    --- CIFAR100
  elseif opt.dataset == 'cifar100' then
    -- Default shortcutType=A and nEpochs=164
    opt.shortcutType = opt.shortcutType == '' and 'A' or opt.shortcutType
    opt.nEpochs = opt.nEpochs == 0 and 164 or opt.nEpochs

  --- Custom Image dataset
  elseif opt.dataset == 'image' then
    opt.nEpochs = opt.nEpochs == 0 and 164 or opt.nEpochs
  else
    cmd:error('unknown dataset: ' .. opt.dataset)
  end

  -- FINETUNING
  if opt.resetClassifier then
    if opt.nClasses == 0 then
      cmd:error('-nClasses required when resetClassifier is set')
    end
  end

  if opt.shareGradInput and opt.optnet then
    cmd:error('error: cannot use both -shareGradInput and -optnet')
  end

  return opt
end

return M
