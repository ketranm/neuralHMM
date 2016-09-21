-- text loader object
-- author: Ke Tran <m.k.tran@uva.nl>
local DataLoader = torch.class('DataLoader')
local moses = require 'moses'

function DataLoader:__init(opt)
    self.unk = '<unk>' -- unknown words for neural model
    self.pad = '<pad>' -- padding
    local trainFile = path.join(opt.dataDir, 'train.txt')
    local trainData = path.join(opt.dataDir, 'train.t7')
    local vocabFile = path.join(opt.dataDir, 'vocab.t7')
    self.mnbz = opt.mnbz

    local maxlen = opt.maxlen  or 40
    self.maxlen = maxlen
    if not path.exists(vocabFile) then
        print('run preprocessing')
        local cutoff = opt.cutoff or 3
        self.vocab = self:makevocab(trainFile, cutoff)
        torch.save(vocabFile, self.vocab)
        self.trainData = self:numberize(trainFile, self.vocab)
        torch.save(trainData, self.trainData)
    else
        self.vocab = torch.load(vocabFile)
        self.trainData = torch.load(trainData)
    end

    self.vocabSize = self.vocab.vocabSize
    self.padidx = self.vocab.w2id[self.pad]
    self.perm = torch.LongTensor()
end

function DataLoader:train()
    local data = self.trainData
    local n = data:size(1)
    local perm = self.perm
    perm:resize(n):randperm(n)
    -- shuffle
    local _data = data:index(1, perm)
    local batches = {}
    return _data:split(self.mnbz, 1)
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

function DataLoader:numberize(textfile, vocab)
    -- create batches
    local w2id = vocab.w2id

    local ids = {}
    -- count number of lines
    local nlines = 0
    for line in io.lines(textfile) do
        local tokens = stringx.split(line)
        if #tokens < self.maxlen then
            nlines = nlines + 1
        end
    end

    local padidx = w2id[self.pad]
    local data = torch.IntTensor(nlines, self.maxlen):fill(padidx)

    local n = 0
    for line in io.lines(textfile) do
        local tokens = stringx.split(line)
        if #tokens < self.maxlen then
            n = n + 1
            for i, tok in ipairs(tokens) do
                data[n][i] = w2id[tok] or w2id[self.unk]
            end
        end
    end

    return data
end

function DataLoader:makevocab(textfile, cutoff)
    local wordFreq = {}
    -- ensure unk and pad in the vocab
    wordFreq[self.unk] = math.huge
    wordFreq[self.pad] = math.huge

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
