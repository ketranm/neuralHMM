-- collections of helpers
local utils = {}

function utils.renorm(x, d)
    local z = nil
    if d then
        z = x:sum(d):add(1e-20)--:expand(#x)
        x:cdiv(z:expand(#x))
    else
        z = x:sum()
        x:div(z)
    end
    return z
end

function utils.scaleClip(v, max_norm)
    local norm = v:norm()
    if norm > max_norm then
        v:div(norm/max_norm)
    end
end

return utils
