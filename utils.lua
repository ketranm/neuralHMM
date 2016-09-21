-- collections of helpers
local utils = {}
local epsilon = 1e-20

function utils.logSumExp(tensor, dim)
    -- aliassing
    local d = dim or tensor:dim()  -- by defaul do it on the last dimension

    local max, maxx = nil, torch.Tensor()
    local n = 0

    if tensor:dim() == 1 then
        local max = tensor:max()
        return torch.log(torch.exp(tensor - max):sum()) + max
    elseif tensor:dim() == 2 then
        assert(d <= 2 and d >= 1)
        max, _ = tensor:max(d)
        n = tensor:size(d)
        if d == 1 then
            maxx:repeatTensor(max, n, 1)
        else
            maxx:repeatTensor(max, 1, n)
        end
        return torch.exp(tensor - maxx):sum(d):log():add(max)
    elseif tensor:dim() == 3 then
        assert(d <= 3 and d >= 1)
        max, _ = tensor:max(d)
        n = tensor:size(d)
        if d == 2 then
            maxx:repeatTensor(max, 1, n, 1)
        elseif d == 3 then
            maxx:repeatTensor(max, 1, 1, n)
        else
            local msg = 'WARNING:\nMost likely you are using it the wrong way.\n' ..
                        'The first dimension usually used for batch size.!'
            print(msg)
            maxx:repeatTensor(max, n, 1, 1)
        end
        return torch.exp(tensor - maxx):sum(d):log():add(max)
    else
        error('only support up to 3D tensor.')
    end
end


function utils.renorm2(tensor, dim)
    -- normalizing tensor along dim
    local d = dim
    local z = nil

    if tensor:dim() == 1 then
        z = tensor:sum()
        tensor:div(z + epsilon)
    elseif tensor:dim() == 2 then
        assert(d <= 2 and d >= 1)
        local n = tensor:size(d)
        if d == 1 then
            z = tensor:sum(d):repeatTensor(n, 1)
        elseif d == 2 then
            z = tensor:sum(d):repeatTensor(1, n)
        end
        z:add(epsilon) -- stablizing
        tensor:cdiv(z)
    elseif tensor:dim() == 3 then
        assert(d <= 3 and d >= 1)
        local n = tensor:size(d)
        if d == 3 then
            z = tensor:sum(d):repeatTensor(1, 1, n)
        elseif d == 2 then
            z = tensor:sum(d):repeatTensor(1, n, 1)
        else
            z = tensor:sum(d):repeatTensor(n, 1, 1)
        end
        z:add(epsilon)
        tensor:cdiv(z)
    else
        error('only support up to 3D tensor.')
    end
    return z
end


function utils.renorm(tensor, dim)
    -- normalizing tensor along dim
    local d = dim
    local z = nil
    tensor:add(epsilon)
    if tensor:dim() == 1 then
        z = tensor:sum()
        tensor:div(z)
    elseif tensor:dim() == 2 then
        assert(d <= 2 and d >= 1)
        local n = tensor:size(d)
        if d == 1 then
            z = tensor:sum(d):repeatTensor(n, 1)
        elseif d == 2 then
            z = tensor:sum(d):repeatTensor(1, n)
        end
        tensor:cdiv(z)
    elseif tensor:dim() == 3 then
        assert(d <= 3 and d >= 1)
        local n = tensor:size(d)
        if d == 3 then
            z = tensor:sum(d):repeatTensor(1, 1, n)
        elseif d == 2 then
            z = tensor:sum(d):repeatTensor(1, n, 1)
        else
            z = tensor:sum(d):repeatTensor(n, 1, 1)
        end
        tensor:cdiv(z)
    else
        error('only support up to 3D tensor.')
    end
    return z
end


function utils.interleave(tensor)
    --[[ this helper is useful for batch training
    assume that input tensor is a (N, T) tensor
    we will need output a table size T of (N, K) matrices
    each entry of the table is feature tensor of all the time step t input tensor
    since there is N input values each time step t, we will need to create a
    flat tensor interleaved of these values before passing it to NN
    ]]

    local interleaved = torch.Tensor(tensor:numel()):typeAs(tensor)
    local N, T = tensor:size(1), tensor:size(2)
    local res = interleaved:split(N, 1)
    for t = 1, T do
        res[t]:copy(tensor[{{}, t}])
    end

    return interleaved
end

function utils.scaleClip(v, max_norm)
    local norm = v:norm()
    if norm > max_norm then
        v:div(norm/max_norm)
    end
end

-- helpers
function utils.diagmat(res, mat)
    -- mat is a (K^2, K) matrix
    -- return (K^2, K^2) matrix
    local n = mat:size(2)
    local m = mat:size(1)
    assert(m == n * n)

    res:resize(m, m):zero()
    for k = 1, m, n do
        res[ {{k, k+n-1}, {k, k+n-1} }] = mat[{{k, k+n-1}, {}}]
    end
    return res
end

function utils.undiagmat(res, mat)
    -- mat is a (K^2, K^2) matrix
    -- return (K^2, K) tensor
    local m = mat:size(1)
    local n = math.sqrt(m)
    res:resize(m, n)
    for k = 1, m, n do
        res[{{k, k+n-1}, {}}] = mat[ {{k, k+n-1}, {k, k+n-1} }]
    end
    return res
end

function utils.eyemat(res, mat)
    local n = mat:size(1)
    res:resize(n^2, n^2)
    for k = 1, n^2, n do
        res[{{k, k+n-1}, {k, k+n-1}}] = mat
    end
    return res
end

return utils
