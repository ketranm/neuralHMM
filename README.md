# neuralHMM

*The code is in cleanup process. We will make all the code availabel around 15/10.*

Code of end to end unsupervised neural HMM. Please make a PR if you want to contribute to the code.

```
Unsupervised Neural Hidden Markov Models  
Ke Tran, Yonatan Bisk, Ashish Vaswani, Daniel Marcu, and Kevin Knight   
EMNLP 2016,  Workshop Structured Prediction for Natural Language Processing
```

## Requirement
To run the code, you need
- Torch
- [lua-utf8](https://github.com/starwing/luautf8), which can be install easily `luarocks install luautf8`
- [cudnn.torch](https://github.com/soumith/cudnn.torch)

## Note
- You can enjoy some speed up if you use `cudnn.LSTM` instead. However, we see better performance when using Justin's LSTM implementation. This is probably due to parameter initialization, which is sensitive in unsupervised learning.

## Data preprocessing
- tokenized, replace all digit by 0 (e.g `tr '0-9' 0`)
- DO NOT lowercase
- one sentence per line

## Acknowledgment
The code is ultilized from
- Justin Johnson's [torch-rnn](https://github.com/jcjohnson/torch-rnn)
- Yoon Kim's [char-lstm](https://github.com/yoonkim/lstm-char-cnn)
