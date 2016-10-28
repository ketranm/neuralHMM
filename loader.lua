-- text loader object
-- author: Ke Tran <m.k.tran@uva.nl>
local DataLoader = torch.class('DataLoader')
local moses = require 'moses'
local utf8 = require 'lua-utf8'


function DataLoader:__init(opt)
    self.unk = '<unk>' -- unknown words for neural model
    self.pad = '<pad>'
    local trainfile = path.join(opt.datapath, 'train.txt')
    local traindata = path.join(opt.datapath, 'train.t7')
    local vocabfile = path.join(opt.datapath, 'vocab.t7')
    self.mnbz = opt.mnbz or 256
    local maxlen = opt.maxlen  or 40
    self.maxlen = maxlen

    local V = opt.vocabsize

    if not path.exists(vocabfile) then
        print('run preprocessing')
        self.vocab = self:makevocab(trainfile, V)
        torch.save(vocabfile, self.vocab)
        self.data = self:numberize(trainfile, self.vocab)
        torch.save(traindata, self.data)
    else
        self.vocab = torch.load(vocabfile)
        self.data = torch.load(traindata)
    end

    self.vocabsize = #self.vocab.idx2word
    assert(self.vocab.idx(self.pad) == 1)
    self.padidx = 1
    self.perm = torch.LongTensor()
end

function DataLoader:train()
    local traindata = {}
    for k, b in ipairs(self.data) do
        local n = b:size(1)
        local perm = self.perm:resize(n):randperm(n)
        local batches = b:index(1, perm):split(self.mnbz, 1)
        for i = 1, #batches do
            local x = batches[i]
            if x:size(1) > self.mnbz / 2 then
                table.insert(traindata, batches[i])
            end
        end
    end

    return traindata
end

function DataLoader:tensorize(str)
    local ws = stringx.split(str)
    local ids = {}
    for _, w in ipairs(ws) do
        table.insert(ids, self.vocab.idx(w))
    end
    return torch.LongTensor(ids)
end

function DataLoader:numberize(textfile, vocab)
    -- count number of lines
    local counter = {}
    local nlines = 0
    for line in io.lines(textfile) do
        local tokens = stringx.split(line, ' ')
        if #tokens < self.maxlen then
            -- bucketing for speed
            local k = math.ceil(#tokens / 20)
            counter[k] = (counter[k] or 0) + 1
            nlines = nlines + 1
        end
    end
    
    local data = {}
    for k, n in pairs(counter) do
        data[k] = torch.IntTensor(n, k*20):fill(1)
    end
    for line in io.lines(textfile) do
        local tokens = stringx.split(line, ' ')
        if #tokens < self.maxlen then
            local k = math.ceil(#tokens / 20)
            local n = counter[k]
            local cur_seq = data[k][n]

            for i, w in ipairs(tokens) do
                cur_seq[i] = vocab.idx(w)
            end

            counter[k] = n-1
        end
    end

    return data
end

function DataLoader:makevocab(textfile, vocabsize)
    local wordfreq = {}
    for line in io.lines(textfile) do
        for w in line:gmatch('%S+') do
            wordfreq[w] = (wordfreq[w] or 0) + 1
        end
    end

    -- sort by frequency
    local words = {}
    for w in pairs(wordfreq) do
        words[#words + 1] = w
    end

    table.sort(words, function(w1, w2)
        return wordfreq[w1] > wordfreq[w2] or
            wordfreq[w1] == wordfreq[w2] and w1 < w2
    end)
    if vocabsize == -1 then
        print('use all words!')
        vocabsize = #words + 2
    else
        vocabsize = math.min(#words + 2, vocabsize)
    end

    local word2idx = {[self.pad] = 1, [self.unk] = 2}
    local idx2word = {self.pad, self.unk}
    for i = 1, vocabsize - 2 do
        local w = words[i]
        table.insert(idx2word, w)
        word2idx[w] = #idx2word
    end

    local vocab = { word2idx = word2idx,
                    idx2word = idx2word,
                    idx = function(w) return word2idx[w] or 2 end,
                    word = function(i) return idx2word[i] end
                    }
    return vocab
end

function DataLoader:getchar(maxlen)
    return self:buildchar(self.vocab.idx2word, maxlen)
end

function DataLoader:buildchar(idx2word, maxlen)
    --[[ Map word to a tensor of character idx
    Parameters:
    - `idx2word`: contiguous table (no hole)
    - `maxlen`: truncate word if is length is excess this threshold
    Returns:
    - `word2char`: Tensor
    ]]
    -- compute max length of words
    local ll = 0 -- longest length
    for _, w in ipairs(idx2word) do
        ll = math.max(ll, utf8.len(w) + 2)
    end
    maxlen = math.min(ll, maxlen)
    print('max word length computed on the corpus: ' .. maxlen)
    -- special symbols
    local char2idx  = {['$$'] = 1}  -- padding
    local idx2char = {'$$'}

    print('create char dictionary!')

    for _, w in ipairs(idx2word) do
        for _, c in utf8.next, w do
            if char2idx[c] == nil then
                idx2char[#idx2char + 1] = c
                char2idx[c] = #idx2char
            end
        end
    end

    local char = {idx2char = idx2char,
                  char2idx = char2idx,
                  padl = #idx2char + 1, padr = #idx2char + 2,
                  idx = function(c) return char2idx[c] end,
                  char = function(i) return idx2char[i] end}

    -- map words to tensor
    char.numberize = function(word, maxlen)
                local x = {char.padl}
                for _, c in utf8.next, word do
                    x[#x + 1] = char.idx(c)
                end
                x[#x + 1] = char.padr
                local out = torch.IntTensor(maxlen):fill(1)
                local x = torch.IntTensor(x)
                if x:numel() < maxlen then
                    out[{{1, x:numel()}}] = x
                else
                    out:copy(x[{{1, maxlen}}])
                end
                return out
            end

    local min = math.min
    local nwords = #idx2word

    local word2char = torch.ones(nwords, maxlen)
    for i, w in ipairs(idx2word) do
        word2char[i] = char.numberize(w, maxlen)
    end

    return word2char
end
