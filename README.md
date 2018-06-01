# implement-of-OLELoss with pytorch
A simple implement of a paper from CVPR2018:OL_'E_ Orthogonal Low-rank Embedding, A Plug and Play Geometric Loss for Deep Learning

The paper has provided a implement on https://github.com/jlezama/OrthogonalLowrankEmbedding .But I find it has some mistakes(or it is my problem).Anyway, I combine its OLELoss function with an example from pytorch's intruduction to get a runable project on my computer.

The IDE I use is Pycharm,and the original OLELoss function raises some yellow warnings.I did try to fix it,but instead I got a lot of erros.But the original one did run well.So if you can correct that please do share it with me or if you have some problem, you can also contact me by email(793472951@qq.com).

# How to use
To use it，you need a datafolder.Note that，you need to set the training set to the form of datafolder/train/class1/…、datafolder/train/class2/… etc.And also do the same to validation set.And change the data_dir in config.py.

Then，you need to create a folder to save weights.The folder take the form like checkpoint in config.py，or you can change it to anyway you like.
