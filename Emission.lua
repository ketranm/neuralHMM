local model, parent = torch.class('nn.EmiNet', 'nn.Module')

function model:__init(nobs, nvars, hidsize)
    local K, V, H = nvars, nobs, hidsize
    self.net = nn.Sequential()
    self.net:add(nn.LookupTable(K, H))
    self.net:add(nn.ReLU())
    self.net:add(nn.Linear(H, H))
    self.net:add(nn.ReLU())
    self.net:add(nn.Linear(H, V))
    self.net:add(nn.LogSoftMax())

    self._input = torch.range(1, K)
    self.gradOutput = torch.Tensor(K, V)
    self._buffer =  torch.Tensor()
end

function model:reset()
    self.net:reset()
end

function model:parameters()
    return self.net:parameters()
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

    if input:type() == 'torch.IntTensor' then
        input = input:long()
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
