# multi-branch_learning_on_conv-tasnet

dependencies for train.py:
torch==1.10.2
soundfile==0.10.3

extra dependencies for eval.py:
pypesq==1.0
pystoi==0.2.2



monaural speech separation:
1)change the dataset path in preprocess.py or preprocess_whamr.py
2)run preprocess.py, create json files which show the file list of datasets
3)change experiment name in nnet/conv_tasnet_mid_feature.py
4)run train.py
5)change experiment name in eval.py
6)run eval.py
