# ReviewNet_text

This project is to build a deep review neural network (DRNN) for text recognition.
The code for attentive lstm and DRNN model will be published in the near future.

The datasets we use are as follows:

Training set (Synth): http://www.robots.ox.ac.uk/~vgg/data/text/

Testing set (ICDAR03): http://pan.baidu.com/s/1i5sPOyT password:3g9n

Testing set (ICDAR13): http://pan.baidu.com/s/1hsOFWu4 password:qdst

Testing set (III5K): http://pan.baidu.com/s/1qY2KZAO password:7uwe

Testing set (SVT): http://pan.baidu.com/s/1gfvBIp1 password:fgus

These datasets include the images, groundtruth, code for lexion generation, different lexions.

We implement our project with Torch7 (http://torch.ch/). If you want to train the model, your computer's GPU RAM should be larger than 2G.

If you want to recognize the scene text with a lexion, you should install pyxDamerauLevenshtein with python. The code is in: https://github.com/gfairchild/pyxDamerauLevenshtein

LstmLayer.lua is the lstm layer with attention model. It is the core of our project.
