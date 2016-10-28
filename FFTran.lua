--[[
Do not go gentle into that good night,
Old age should burn and rave at close of day;
Rage, rage against the dying of the light.
]]

local model, parent = torch.class('nn.FFTran', 'nn.Module')
function model:__init(nvars, hidsize)
    -- alias
    self.nvars = nvars
    local K = nvars
    local H = hidsize
    self.net = nn.Sequential()
    self.net:add(nn.Linear(1, H, false))
    self.net:add(nn.ReLU())
    self.net:add(nn.Linear(H, H, false))
    self.net:add(nn.ReLU())
    self.net:add(nn.Linear(H, K^2))
    self.net:add(nn.View(-1, K))
    self.net:add(nn.LogSoftMax())

    self._input = torch.ones(1)

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
    local K = self.nvars
    local N, T = input:size(1), input:size(2)
    if self._cache then
        self._logp = self._cache
    else
        self._logp = self.net:forward(self._input)
    end
    return self._logp:view(1, 1, K, K):expand(N, T, K, K)
end

function model:update(input, gradOutput)
    self.net:backward(self._input, gradOutput:sum(1):sum(2):squeeze())
end
