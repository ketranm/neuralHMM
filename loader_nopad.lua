-- text loader object
-- author: Ke Tran <m.k.tran@uva.nl>
local DataLoader = torch.class('DataLoader')
local moses = require 'moses'

function DataLoader:__init(opt)
    self.unk = '<unk>' -- unknown words for neural model

    local trainFile = path.join(opt.dataDir, 'train.txt')
    local trainData = path.join(opt.dataDir, 'train.t7')
    local vocabFile = path.join(opt.dataDir, 'vocab.t7')

    local mnbz = opt.mnbz or 20 -- size of minibatch
    local maxlen = opt.maxlen  or 40
    self.maxlen = maxlen
    if not path.exists(vocabFile) then
        print('run preprocessing')
        local cutoff = opt.cutoff or 3
        self.vocab = self:makevocab(trainFile, cutoff)
        torch.save(vocabFile, self.vocab)
        self.trainData = self:numberize(trainFile, self.vocab, mnbz)
        torch.save(trainData, self.trainData)
    else
        self.vocab = torch.load(vocabFile)
        self.trainData = torch.load(trainData)
    end

    self.vocabSize = self.vocab.vocabSize
    self.eosidx = self.vocab.w2id[self.eos]
end

function DataLoader:train()
    return moses.shuffle(self.trainData)
end

function DataLoader:tensorize(str)
    local w2id = self.vocab.w2id
    local tokens = stringx.split(str)
    local ids = {}
    for _, w in ipairs(tokens) do
        table.insert(ids, w2id[w] or w2id[self.unk])
    end
    return torch.LongTensor(ids)
end

function DataLoader:numberize(textfile, vocab, mnbz)
    -- create batches
    local w2id = vocab.w2id
    local shard = {}
    local ids = {}

    for line in io.lines(textfile) do
        local tokens = stringx.split(line)
        if #tokens <= self.maxlen then
            local len = #tokens
            if not shard[len] then shard[len] = {} end
            local _shard = shard[len]
            for _, w in ipairs(tokens) do
                table.insert(_shard, w2id[w] or w2id[self.unk])
            end
        end
    end

    local data = {}
    for k, _shard in pairs(shard) do
        local shard_k = torch.IntTensor(_shard):split(mnbz * k, 1)
        for _, x in ipairs(shard_k) do
            local y = x:view(-1, k)
            if y:size(1) > 1 then
                -- ignoring singleton
                table.insert(data, x:view(-1, k))
            end
        end
    end

    return data
end

function DataLoader:makevocab(textfile, cutoff)
    local wordFreq = {}
    -- ensure unk and pad in the vocab
    wordFreq[self.unk] = math.huge

    for line in io.lines(textfile) do
        for w in line:gmatch('%S+') do
            wordFreq[w] = (wordFreq[w] or 0) + 1
        end
    end

    local n = 0
    local w2id, id2w = {}, {}
    for w, c in pairs(wordFreq) do
        if c > cutoff then
            n = n + 1
            w2id[w] = n
            id2w[n] = w
        end
    end

    return {w2id = w2id, id2w = id2w, vocabSize = n}
end
