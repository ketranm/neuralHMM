--[[ Message Passing for the 1st order HMM (linear chain model)
Reference:
    [1] A tutorial on hidden Markov models and selected applications
            in speech recognition.
            L. R. Rabiner ; AT&T Bell Lab., Murray Hill, NJ, USA
    [2] http://www.cs.cmu.edu/~roni/11761-s16/assignments/shen_tutorial.pdf

Author: Ke Tran <m.k.tran@uva.nl>

NOTE: This code is written for GPUs, and for the love of the speed.
For that reason, I intentinally use scaling-factor instead of sumlogexp when
computing alpha and beta messages (multiplying is much faster).
The downside of using scaling factor is that, sometimes, we can not compute
exactly those messages due to numerical approximation.
This often happens at the begining of learning.
Nevertheless, when I tested (-set debug = true)
the tolerance limit of approximation is acceptable.
]]

local BaumWelch, parent = torch.class('nn.BaumWelch', 'nn.Module')
local utils = require 'utils'

function BaumWelch:__init(padidx)
    self.padidx = padidx
    -- for message passing algorithm
    self.alpha  = torch.Tensor()
    self.beta   = torch.Tensor()
    self.gamma  = torch.Tensor()
    self.eta    = torch.Tensor()
    self.scale = torch.Tensor()

    -- BUFFER TENSOR
    self.prob_trans = torch.Tensor()
    self.prob_emiss = torch.Tensor()
    self.prob_prior = torch.Tensor()
    self.buffer = torch.Tensor()

    self.debug = true
end


function BaumWelch:run(input, stats)
    local N = input:size(1) -- batch size
    local T = input:size(2) -- sequence length
    local buffer = self.buffer
    local prob_prior = self.prob_prior
    local prob_trans = self.prob_trans
    local prob_emiss = self.prob_emiss

    local log_emiss, log_trans, log_prior = unpack(stats)

    local K = log_prior:numel()

    prob_prior:resizeAs(log_prior):copy(log_prior):exp()
    prob_trans:resizeAs(log_trans):copy(log_trans):exp()
    prob_emiss:resizeAs(log_emiss):copy(log_emiss):exp()

    local masked = input:ne(self.padidx)
    -- we need this for computing correctly the log-likelihood
    local masked_pad = input:eq(self.padidx)

    -- Message Passing

    -- nicely alias
    local alpha = self.alpha
    local beta  = self.beta
    local gamma = self.gamma
    local scale = self.scale
    local eta   = self.eta

    -- FORWARD MESSAGE

    alpha:resize(N, T, K):zero()
    scale:resize(N, T):zero()

    -- (1) compute the first alpha
    local a1 = alpha[{{}, 1, {}}]
    a1:add(prob_prior:view(1, -1):expand(N, K))
    a1:cmul(prob_emiss[{{}, 1}])

    -- rescale
    scale[{{}, {1}}] = utils.renorm(alpha[{{}, 1}], 2)

    -- (2) compute the rest of alpha
    for t = 2, T do
        local emi_t = prob_emiss[{{}, {t}}]
        local curr_a = alpha[{{}, {t}}] -- (N, 1, K)
        local prev_a = alpha[{{}, {t-1}}]
        local tran_t = prob_trans[{{}, t-1}]
        -- transition matrix is row major, sum over row should return 1
        curr_a:bmm(prev_a, tran_t):cmul(emi_t)
        scale[{{}, {t}}] = utils.renorm(alpha[{{}, t}], 2)
    end

    -- BACKWARD MESSAGE

    beta:resize(N, T, K):fill(1)

    -- because we store inverted of scaling factor
    beta[{{}, T}]:cdiv(scale[{{}, {T}}]:expand(N, K))
    buffer:resize(N, 1, K)

    -- NOTE: if pad is in the last index, we need to overwrite beta
    -- check boundary of sequence where pad appear for the first time
    -- eos: is used to check the true eos, after this point, pad appears
    -- because pad is always at the end of the sequence,
    -- we will do it in one go

    local eos = masked[{{}, {1, T-1}}]:ne(masked[{{}, {2, T}}])
    for t = T-1, 1, -1 do
        local eos_t = eos[{{}, {t}}]:expand(N, K)
        local emi_t = prob_emiss[{{}, {t+1}}]
        local prev_b = beta[{{}, {t+1}}]
        local curr_b = beta[{{}, {t}}]
        buffer:cmul(prev_b, emi_t)
        local tran_t = prob_trans[{{}, t}]
        curr_b:bmm(buffer, tran_t:transpose(2, 3))
        if eos_t:sum() > 0 then
            curr_b:maskedFill(eos_t, 1)
        end
        curr_b:cdiv(scale[{{}, {t}}]:expand(N, K))
    end

    -- compute posteriors
    -- NOTE: the beta message is computed correctly up to EOS symbols
    -- after that, it's incorrect, but we keep it for the sake of speed
    -- the gamma is correctly computed, we will use masked_w to zero out
    -- EOS symbols

    gamma:resize(N, T, K):zero()
    for t = 1, T do
        local gamma_t = gamma[{{}, {t}}]
        gamma_t:cmul(alpha[{{}, {t}}], beta[{{}, {t}}])
        -- NOTE un-comment for debugging purpose
        --[[
        gamma_t:cmul(scale[{{}, {t}}]:expand(N, K)) -- sweet eq(110), partial term
        if self.debug then
            -- checking correctness p(z | s): \sum_z p(z | s) = 1
            local checksum = gamma[{{}, t}]:sum(2):add(-1):cmul(masked[{{}, t}]):abs():sum()
            assert(checksum < 1e-2, string.format('gamma checksum error %.7f', checksum))
        end
        ]]
    end
    -- comment out the following line (renorm) in debugging mode
    utils.renorm(gamma, 3)
    masked = masked:double()  -- convert to double tensor
    gamma:cmul(masked:view(N, T, 1):expand(N, T, K))

   --[[ Compute eta
    Now we compute eta. The eta is only available from the begining of sequence
    to the index before the real end of sequence
    for example, if 0 is used to indicate EOS then
    input = 1 2 3 4 5 0 0 0
    the etas are only needed for indices 1 2 3 4
    which are corresponding to transition from (1, 2), (2, 3), ..., (4, 5)
    so to compute the correct eta at time step t, we need to know whether
    word at time step t+1 is a EOS or not.
    ]]

    eta:resize(N, T, K, K):zero()
    for t = 1, T-1 do
        local emi_t = prob_emiss[{{}, {t+1}}]
        local bmsg = beta[{{}, {t+1}}]
        local amsg = alpha[{{}, {t}}]
        local tran_t =  prob_trans[{{}, t}]
        local eta_t = eta[{{}, t}]
        -- NOTE: un-comment for debugging
        --[[
        bmsg:cmul(emi_t):cmul(masked[{{}, {t+1}}]:expand(N, K))
        eta_t:bmm(amsg:transpose(2, 3), bmsg):cmul(tran_t)

        if self.debug then
            -- this is what happened: when we see the real symbol before padding (EOS) at time t
            -- the p(z_t | x) exists but p(z_t, z_{t+1}| x)
            -- so we have to zero out p(z_t | x) when do checking
            local eos_t= eos[{{}, {t}}] -- check for end of sequence
            local gamma_t = gamma[{{}, t}]
            local derr = eta_t:sum(3):squeeze():add(-1, gamma_t)
            derr:maskedFill(eos_t:expand(N, K), 0)
            local checksum = derr:abs():sum()
            assert(checksum < 1e-3, string.format('eta checksum error %.7f', checksum))
            -- good job Ke! This is pain in the ass.
        end
        ]]
        -- comment out the following lines in debugging mode
        bmsg:cmul(emi_t)
        eta_t:bmm(amsg:transpose(2, 3), bmsg):cmul(tran_t) -- will be N, K, K
        local z = eta_t:sum(2):sum(3):expand(N, K, K)
        eta_t:cdiv(z):cmul(masked[{{}, {t+1}}]:contiguous():view(N, 1, 1):expand(N, K, K))
        --eta_t:cdiv(z):cmul(masked[{{}, {t}}]:contiguous():view(N, 1, 1):expand(N, K, K))
    end

    local prior = gamma[{{}, 1}]:sum(1):squeeze()

    scale:maskedFill(masked_pad, 1)
    local loglik = scale:clone():log():sum() / masked:sum()
    -- return posteriors
    return {prior, eta, gamma}, loglik
end

function BaumWelch:argmax(input, stats)
    -- inference, we just need alpha message
    local T = input:numel()
    local prob_prior = self.prob_prior
    local prob_trans = self.prob_trans
    local prob_emiss = self.prob_emiss

    local log_emiss, log_trans, log_prior = unpack(stats)

    local K = log_prior:numel()

    prob_prior:resizeAs(log_prior):copy(log_prior):exp()
    prob_trans:resizeAs(log_trans):copy(log_trans):exp()
    prob_emiss:resizeAs(log_emiss):copy(log_emiss):exp()

    local alpha = self.alpha
    alpha:resize(T, K):zero()

    -- borrow viterbi-path implementation from Kevin Murphy
    -- psi[t][j]: the best predecessor state,
    -- given that we ended up in state j at t
    local psi = torch.zeros(T, K):typeAs(alpha)

    local a1 = alpha[{1, {}}]
    a1:add(prob_prior)
    a1:cmul(prob_emiss[{{}, 1}])
    utils.renorm(alpha[{{1}, {}}], 2)

    for t = 2, T do
        local emi_t = prob_emiss[{{}, t}]
        local curr_a = alpha[t]
        local prev_a = alpha[t-1]
        local z = prev_a:view(-1, 1):repeatTensor(1, K)
        z:cmul(prob_trans[{{}, t-1}])
        local val, idx = z:max(1)
        psi[t]:copy(idx)
        curr_a:copy(val)
        curr_a:cmul(emi_t)
        utils.renorm(curr_a, 1)
    end
    local val, idx = alpha[{T, {}}]:max(1)
    local path = torch.zeros(T)
    path[T] = idx[1]
    for t = T-1, 1, -1 do
        path[t] = psi[t+1][path[t+1]]
    end

    return path
end
