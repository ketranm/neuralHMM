local model, parent = torch.class('nn.Prior', 'nn.Module')

function model:__init(nvars)
    self.nvars = nvars
    self.net = nn.Sequential()
    self.net:add(nn.Linear(1, nvars))
    self.net:add(nn.LogSoftMax())

    self.proto = torch.ones(1)
end


function model:parameters()
    return self.net:parameters()
end

function model:training()
    self.net:training()
    parent.training(self)
end

function model:evaluate()
    self.net:evaluate()
    parent.evaluate(self)
end

function model:reset()
    self.net:reset()
end

function model:precompute()
    self._cache = self.net:forward(self.proto)
end

function model:log_prob(input)
    if self._cache then
        return self._cache
    else
        return self.net:forward(self.proto)
    end
end

function model:update(input, gradOutput)
    self.net:backward(self.proto, gradOutput)
end
