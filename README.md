# neuralHMM

Code of end to end unsupervised neural HMM described in our paper: https://arxiv.org/abs/1609.09007

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
$ th main.lua -datapath ../wsj -nstates 45 -niters 20 \
 -hidsize 512 -mnbz 256 -nloops 6 -cuda -maxlen 81 \
 -nlayers 3 -modelpath ../save -model lstmconv -conv -lstm
```

The option `-lstm` indicates using LSTM transition, and `-conv` indicates using Char-CNN emission.

This code is heavily optimized, you can train it using very long sentences. `-maxlen` option is used to specify maximum sentence length you want to use. The loader will automatically re-arrange data in to batches of length 20, 40, 60, and 80 (when `-maxlen 81` is on).

For evaluation, assume that you have a trained model file `../save/nhmm.iter19.t7`, you can perform tagging by

```
$ th main.lua -datapath ../wsj -nstates 45 -hidsize 512 \
-nlayers 3 -cuda -model ../save/nhmm.iter19.t7 -conv -lstm \
-input your_text_file.txt -output -tagged_file.txt
```

If `-cuda` is provided, the model will use GPU. For the Char-CNN (`-conv` option), you have to use GPU.

### Bell and Whistle
#### Probing
Since unsupervised models are sensitive to initialization, ideally you would like to run a dozen (or even hundreds) instances of the NHMM and pick the best instance based on likelihood for testing. This is rather expensive. We find that the final likelihood (at the last iteration) highly correlates with the likelihood at the first iteration among different runs. Therefore, we introduce a trick called **probing**. Essentially, we will run the first iteration for `nprobes` times. Each time, all the parameters are reset and we use different random seeds. After `nprobes` runs, we pick the run that gives the highest log-likihood to continue. Setting `nprobes` to 5 usually produces good results.

**NOTE**: We didn't use this trick in our paper. If you do not have development data to evaluate your model, using probing is strongly advised.

#### Adding noise
Another trick to get good performance is adding gradient noise (https://arxiv.org/abs/1511.06807). We didn't use this trick in our paper, but empirically we find that adding gradient noise often gives stable and good performance.

## Evaluation
The evaluation script was prepared by [Yonatan Bisk](http://yonatanbisk.com/). It's quite self contained.  It expects data in the format of one sentence per line with tags whitespace separated.

```
$ python eval.py predicted_tagged.txt gold_tagged.txt
```

## TODO:
We use cutorch 1.0 in our experiments. Need to make this code compatible for last cutorch.
- Map corresponding cpu tensors to CUDA tensors

## Acknowledgment
The code is utilized from
- Justin Johnson's [torch-rnn](https://github.com/jcjohnson/torch-rnn)
- Yoon Kim's [char-lstm](https://github.com/yoonkim/lstm-char-cnn)
