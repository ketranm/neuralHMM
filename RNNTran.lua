require 'LSTM'
local model, parent = torch.class('nn.RNNTran', 'nn.Module')
function model:__init(nobs, nvars, hidsize, nlayers, dropout)
    self.nvars = nvars
    local K = self.nvars
    self.dropout = dropout
    local V, D, H = opt.nobs, hidsize, hidsize

    self.net = nn.Sequential()
    self.rnns = {}
    self.net:add(nn.LookupTable(V, D))
    for i = 1, nlayers do
        local prev_dim = H
        if i == 1 then prev_dim = D end
        local  rnn = nn.LSTM(prev_dim, H)
        rnn.remember_states = false
        table.insert(self.rnns, rnn)

        self.net:add(rnn)
        if self.dropout > 0 then
          self.net:add(nn.Dropout(self.dropout))
        end
    end

    self.net:add(nn.View(-1, H))
    self.net:add(nn.Linear(H, K^2)) -- N * T, K^2
    self.net:add(nn.View(-1, K)) -- N * T * K, K
    self.net:add(nn.LogSoftMax())
    self.viewx = nn.View()
    self.net:add(self.viewx)
end

function model:parameters()
    return self.net:parameters()
end

function model:precompute()
end

function model:training()
    self.net:training()
    parent.training(self)
end

function model:evaluate()
    self.net:evaluate()
    parent.evaluate(self)
end

function model:log_prob(input)
    local N, T = input:size(1), input:size(2)
    local K = self.nvars
    self.viewx:resetSize(N, T, K, K)
    return self.net:forward(input) -- will be (N * T, K, K)
end


function model:update(input, gradOutput)
    self.net:backward(input, gradOutput)
    self:resetStates()
end


function model:resetStates()
    for i, rnn in ipairs(self.rnns) do
        rnn:resetStates()
    end
end
