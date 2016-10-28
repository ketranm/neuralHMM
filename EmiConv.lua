local factory = require 'factory'

local model, parent = torch.class('nn.EmiConv', 'nn.Module')
function model:__init(word2char, nvars, feature_maps, kernels, charsize, hidsize)
    local K = nvars
    self.word2char = word2char
    local char_dim = charsize
    local H = hidsize
    local featsize = torch.Tensor(feature_maps):sum()
    local nchars = word2char:max()
    local maxchars = word2char:size(2)
    local V = word2char:size(1)

    local char_cnn = factory.build_cnn(feature_maps, kernels, charsize, hidsize, nchars, maxchars)

    local state_emb = nn.Sequential()
    state_emb:add(nn.LookupTable(K, H))
    state_emb:add(nn.ReLU())

    local prl = nn.ParallelTable()
    prl:add(state_emb)
    prl:add(char_cnn)

    local bias = nn.Linear(1, V, false) -- shared

    local emi0b = nn.Sequential()
    emi0b:add(prl)
    emi0b:add(nn.MM(false, true))

    local prlx = nn.ParallelTable()
    prlx:add(emi0b)
    prlx:add(bias)

    local emi = nn.Sequential()
    emi:add(prlx)
    emi:add(nn.CAddTable())
    emi:add(nn.LogSoftMax())
    self.net = emi

    self._input = {{torch.range(1, K), self.word2char}, torch.ones(K, 1)}
    self.gradOutput = torch.Tensor(K, V)
    self._buffer =  torch.Tensor()
end

function model:reset()
    self.net:reset()
end

function model:training()
    self.net:training()
    parent.training(self)
end

function model:evaluate()
    self.net:evaluate()
    parent.evaluate(self)
end

function model:precompute()
    self._cache = self.net:forward(self._input)
end

function model:log_prob(input)
    local N, T = input:size(1), input:size(2)
    if not self._cache then
        self._logp = self.net:forward(self._input)
    else
        self._logp = self._cache
    end

    return self._logp:index(2, input:view(-1)):view(-1, N, T):transpose(1, 2):transpose(2, 3)
end

function model:update(input, gradOutput)
    local N, T = input:size(1), input:size(2)
    local dx = gradOutput:transpose(2, 3):transpose(1, 2)
    self._buffer:resizeAs(dx):copy(dx)
    self.gradOutput:zero()
    self.gradOutput:indexAdd(2, input:view(-1), self._buffer:view(-1, N * T))
    self.net:backward(self._input, self.gradOutput)
end

function model:parameters()
    return self.net:parameters()
end
