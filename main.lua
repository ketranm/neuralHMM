local cmd = torch.CmdLine()
cmd:text('unsupervised HMM')
cmd:text('Data')
cmd:option('-datapath', '../data', 'location of data')
cmd:option('-vocabsize', -1, 'size of dynamic softmax, -1 for using all')
cmd:option('-mnbz', 250, 'size of minibatch')
cmd:option('-maxlen', 40, 'maximum number of tokens per sentence')

cmd:text('model')
cmd:option('-hidsize', 128, 'hidden size of char-softmax, use when -use_char is true')
cmd:option('-maxchars', 15, 'use char-softmax')
cmd:option('-kernels', {1,2,3,4,5,6,7}, 'kernels of char-cnn')
cmd:option('-feature_maps', {50, 100, 128, 128, 128, 128, 128}, 'feature map of char-cnn')
cmd:option('-charsize', 15, 'char embedding dim')
cmd:option('-nstates', 45, 'number of latent states')
cmd:option('-conv', false, 'use Char-CNN for emission')
cmd:option('-lstm', false, 'use LSTM for transition')
cmd:option('-max_word_len', 15, 'truncate word that is longer than this, use for Char-CNN')
cmd:option('-nlayers', 3, 'number of lstm layers')

cmd:text('optimization')
cmd:option('-nloops', 6, 'number of inner loops for optim when -nn is true')
cmd:option('-niters', 16, 'number of iterations/epochs')
cmd:option('-max_norm', 5, 'max gradient norm')
cmd:option('-dropout', 0.5, 'dropout')
cmd:option('-report_every', 10, 'print out after certain number of mini batches')
cmd:option('-modelpath', '../cv', 'saved model location')

cmd:text('utils')
cmd:option('-model', 'hmm', 'trained model model file')
cmd:option('-output', '../data/pred.txt', 'output prediction')
cmd:option('-input', '', 'input file to predict')

cmd:option('-cuda', false, 'using cuda')
cmd:option('-debug', false, 'debugging mode')

cmd:text()
opt = cmd:parse(arg or {})

print(opt)
--torch.manualSeed(42)
require 'nn'
require 'BaumWelch'
require 'loader'
if opt.cuda then
    require 'cunn'
    require 'cutorch'
    print('using GPU')
    --cutorch.manualSeed(42)
end

-- loading data
local loader = DataLoader(opt)
print('vocabulary size: ', loader.vocabsize)
-- add number of words to opt
opt.nobs = loader.vocabsize
opt.padidx = loader.padidx or -1
opt.vocab = loader.vocab

require 'optim'
local model_utils = require 'model_utils'
local utils = require 'utils'

print('create networks')
require 'Prior'
require 'FFTran'

local prior_net = nn.Prior(opt.nstates)
local emiss_net, trans_net
if opt.conv then
    print('use Convolutional Character Model')
    require 'EmiConv'
    local word2char = loader:getchar(opt.max_word_len)
    emiss_net = nn.EmiConv(word2char, opt.nstates, opt.feature_maps, opt.kernels, opt.charsize, opt.hidsize)
    print('set up Char-CNN completed!')
else
    print('use Feed-forward Emission Model')
    require 'Emission'
    emiss_net = nn.EmiNet(opt.nobs, opt.nstates, opt.hidsize)
end

if opt.lstm then
    print('use LSTM for transition')
    require 'RNNTran'
    trans_net = nn.RNNTran(opt.nobs, opt.nstates, opt.hidsize, opt.nlayers, opt.dropout)
else
    trans_net = nn.FFTran(opt.nstates, opt.hidsize)
end

local inference = nn.BaumWelch(opt.padidx)

if opt.cuda then
    prior_net:cuda()
    emiss_net:cuda()
    trans_net:cuda()
    inference:cuda()
end

prior_net:reset()
emiss_net:reset()
trans_net:reset()

local params, gradParams
    = model_utils.combine_all_parameters(emiss_net, trans_net, prior_net)

-- It seems that uniform initialization works the best for feed-forward model
--params:uniform(-1e-3, 1e-3)

function process(input)
    -- keep in mind that we do padding
    -- so for each batch, we will take the length as the lenth
    -- of the max sequence without pad
    local real_length = input:size(2)
    local mnbz = input:size(1)
    for i = real_length, 1, -1 do
        if input[{{}, i}]:sum() > mnbz then
            real_length = i
            break
        end
    end
    return input[{{}, {1, real_length}}]
end


function train()
    trans_net:training()
    emiss_net:training()
    prior_net:training()
    local optim_config, optim_states = {}, {}
    local nprobes = 5
    local best_start_loglik = -1000
    local iter = 0
    -- adding noise to gradient
    local gnoise = {}
    gnoise.t = 0
    gnoise.noise = gradParams.new()
    gnoise.noise:resizeAs(gradParams)
    gnoise.tau = 0.01
    gnoise.gamma = 0.55

    while iter <  opt.niters do
        local loglik = 0
        local data = loader:train()
        for j = 1, #data do
            local input = process(data[j])
            if opt.cuda then
                input = input:cuda()
            else
                input = input:long()
            end
            local count, f = nil, nil
            local prev_f = nil

            for k = 1, opt.nloops do
                local log_prior = prior_net:log_prob(input)
                local log_trans = trans_net:log_prob(input)
                local log_emiss = emiss_net:log_prob(input)
                local stats = {log_emiss, log_trans, log_prior}

                count, f = inference:run(input, stats)
                if not prev_f then
                    prev_f = f
                else
                    local improve = f - prev_f
                    local imp = -improve / prev_f -- note that f is negative
                    if imp < 1e-4 then
                        break
                    else
                        prev_f = f
                    end
                end

                -- update
                local feval = function(x)
                    gradParams:zero()
                    if params ~= x then params:copy(x) end
                    prior_net:update(input, count[1]:mul(-1 / opt.nstates))
                    trans_net:update(input, count[2]:mul(-1 / opt.nstates))
                    emiss_net:update(input, count[3]:mul(-1 / opt.nstates))

                    --gradParams:add(1e-3, params)
                    utils.scaleClip(gradParams, 3)

                    -- gradient noise
                    local var = gnoise.tau / torch.pow(1 + gnoise.t, gnoise.gamma)
                    gnoise.noise:normal(0, var)
                    gradParams:add(gnoise.noise)
                    gnoise.t = gnoise.t + 1

                    return _, gradParams
                end

                optim.adam(feval, params, optim_config, optim_states)
            end
            loglik = loglik + f
            if j % opt.report_every == 0 then
                io.write(string.format('iter %d\tloglik %.4f\t %.3f\r', iter, loglik/j, j/#data))
                io.flush()
                collectgarbage()
            end
        end

        local curr_loss = loglik/#data
        local modelfile = string.format('%s/%s.iter%d.t7', opt.modelpath, opt.model, iter)
        if nprobes > 0 then
            print(string.format('current loss %.3f\tbest %.3f\tremained probes %d', curr_loss, best_start_loglik, nprobes - 1))
            if curr_loss > best_start_loglik then
                best_start_loglik = curr_loss
                paths.mkdir(paths.dirname(modelfile))
                print(string.format('probe: %d loglik %.4f ||| states: %s', nprobes, curr_loss, modelfile))
                local probe_states = {params = params, optim_config = optim_config, optim_states = optim_states, t = gnoise.t}
                torch.save(modelfile, probe_states)
            end
            nprobes = nprobes - 1
            -- reseeding
            torch.seed()
            cutorch.seed()
            optim_config = {}
            optim_states = {}
            gnoise.t = 0
            prior_net:reset()
            trans_net:reset()
            emiss_net:reset()
        elseif nprobes == 0 then
            -- load file, recover all optimization states
            print('end of probing, use the best probing model to continue training!')
            local probe_states = torch.load(modelfile)
            params:copy(probe_states.params)
            optim_config = probe_states.optim_config
            optim_states = probe_states.optim_states
            gnoise.t = probe_states.t
            nprobes = -1
            iter = iter + 1
        else
            paths.mkdir(paths.dirname(modelfile))
            torch.save(modelfile, params)
            print(string.format('saved: %s\tloglik  %.4f\titer %d', modelfile, curr_loss, iter))
            iter = iter + 1
        end
    end
end


function infer(textfile, predfile, modelfile)
    -- batch inference
    print(string.format('load model: %s', modelfile))
    params:copy(torch.load(modelfile))
    emiss_net:precompute()
    trans_net:evaluate()
    emiss_net:evaluate()
    prior_net:evaluate()
    prior_net:precompute()
    local fw = io.open(predfile, 'w')
    local n = 0
    local sents = {}
    for line in io.lines(textfile) do
        sents[#sents + 1] = loader:tensorize(line):view(1, -1)
    end
    local n = #sents
    local mnbz = 256
    for i = 1, n, mnbz do
        local max_seq_len = 0
        local bs = 0
        for k = i, math.min(i + mnbz - 1, n) do
            if sents[k]:numel() > max_seq_len then max_seq_len = sents[k]:numel() end
            bs = bs + 1
        end
        local input = torch.IntTensor(bs, max_seq_len):fill(1)
        for k = 0, bs - 1 do
            local x = sents[k+i]
            input[{{k+1}, {1, x:numel()}}] = x
        end

        input = input:cuda()
        io.write(string.format('sent: %d\r', i))
        io.flush()
        local log_prior = prior_net:log_prob(input)
        local log_trans = trans_net:log_prob(input)
        local log_emiss = emiss_net:log_prob(input)

        for k = 0, bs - 1 do
            local x = sents[k+i]
            local ex = log_emiss[{{k + 1}, {1, x:numel()}}]
            local tx = log_trans[{{k + 1}, {1, x:numel()}}]

            local predx = inference:argmax(x, {ex, tx, log_prior})
            local output = table.concat(torch.totable(predx), ' ')
            fw:write(output .. '\n')
        end
    end
    fw:close()
end


--- main script

if opt.input ~= '' then
    infer(opt.input, opt.output, opt.model)
else
    train()
end
