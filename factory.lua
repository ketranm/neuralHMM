-- quickly build char-CNN
require 'cudnn'
require 'nngraph'
local factory = {}

function factory.highway(size, num_layers, bias, f)
    -- size = dimensionality of inputs
    -- num_layers = number of hidden layers (default = 1)
    -- bias = bias for transform gate (default = -2)
    -- f = non-linearity (default = ReLU)
    -- seem blows up memory
    -- TODO: not using nngraph
    local output, transform_gate, carry_gate
    local num_layers = num_layers or 1
    local bias = bias or -2
    local f = f or nn.ReLU()
    local input = nn.Identity()()
    local inputs = {[1]=input}
    for i = 1, num_layers do
        output = f(nn.Linear(size, size)(inputs[i]))
        transform_gate = nn.Sigmoid()(nn.AddConstant(bias)(nn.Linear(size, size)(inputs[i])))
        carry_gate = nn.AddConstant(1)(nn.MulConstant(-1)(transform_gate))
        output = nn.CAddTable()({
           nn.CMulTable()({transform_gate, output}),
           nn.CMulTable()({carry_gate, inputs[i]})})
        table.insert(inputs, output)
    end
    return nn.gModule({input},{output})
end

function factory.build_cnn(feature_maps, kernels, charsize, hidsize, nchars, maxlen)
    local featsize = torch.Tensor(feature_maps):sum()
    local net = nn.Sequential()
    net:add(nn.LookupTable(nchars, charsize, 1))
    local concat = nn.ConcatTable()
    for i = 1, #kernels do
        local reduced_l = maxlen - kernels[i] + 1
        local conv = cudnn.SpatialConvolution(1, feature_maps[i], charsize,
                                            kernels[i], 1 , 1, 0)
        local view = nn.View(1, -1, charsize):setNumInputDims(2)

        local inet = nn.Sequential()
        inet:add(view)
        inet:add(conv)
        inet:add(cudnn.Tanh())
        inet:add(cudnn.SpatialMaxPooling(1, reduced_l, 1, 1, 0, 0))
        inet:add(nn.Squeeze())
        concat:add(inet)
    end
    net:add(concat)
    net:add(nn.JoinTable(2))
    net:add(nn.View(-1, featsize))
    net:add(nn.Linear(featsize, hidsize))
    return net
end

return factory
