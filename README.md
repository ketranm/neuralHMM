# neuralHMM


**The code is in cleanup process. We will make all the code available around October 15.**

Code of end to end unsupervised neural HMM.

```
Unsupervised Neural Hidden Markov Models  
Ke Tran, Yonatan Bisk, Ashish Vaswani, Daniel Marcu, and Kevin Knight   
Workshop Structured Prediction for Natural Language Processing, EMNLP 2016
```

## Requirement
To run the code, you need
- Torch
- [lua-utf8](https://github.com/starwing/luautf8), which can be installed easily `luarocks install luautf8`
- [cudnn.torch](https://github.com/soumith/cudnn.torch)

## Note
- You can enjoy some speed up if you use `cudnn.LSTM` instead. However, we see better performance when using Justin's LSTM implementation. This is probably due to parameter initialization, which is sensitive in unsupervised learning.

## Code Structure
This code is inspired by [Edward](http://edwardlib.org/). Each Model file (e.g `EmiConv.lua`, `FFTran.lua`) has API `log_prob` that returns the log probabilities. These log probabilities of prior, emission, and transition then will be passed to inference module (BaumWelch) that computes the posteriors.


## Data preprocessing
- tokenized, replace all digit by 0 (e.g `tr '0-9' 0`)
- DO NOT lowercase
- one sentence per line

## Running the script

File `main.lua` is the main script to train and evaluate an unsupervised neural Hidden Markov Model.

To train the full version of NHMM + Convlutional Character Emission + LSTM Transition, run

```
$ th main.lua -datapath ../wsj -nstates 45 -niters 20 -hidsize 512 -mnbz 256 -nloops 6 -cuda -modelpath ../save -model lstmconv -conv -lstm
```

The option `-lstm` indicates using LSTM transition, and `-conv` indicates using Char-CNN emission.


## Acknowledgment
The code is ultilized from
- Justin Johnson's [torch-rnn](https://github.com/jcjohnson/torch-rnn)
- Yoon Kim's [char-lstm](https://github.com/yoonkim/lstm-char-cnn)
