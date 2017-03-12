# SRCNN-keras-hdf5
1. 本repo基于keras框架与SRCNN网络，利用HDF5库能够读写超过内存的大数据的特点，写了个能够直接用于超过内存的大数据的SRCNN demo。

2. demo重点展示了hdf5的分块读写代码与应用思路，使之能够直接应用于超过内存的大数据

3. demo 参考了 http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html ，训练数据可以从链接中直接下载，下载后放置于根目录下的
/dataset/Train 路径中

4. 关于hdf5读写超过内存的大数据的方法与技巧，我在这篇博客中做了少许说明 

  http://www.cnblogs.com/nwpuxuezha/p/6537307.html

5. 欢迎提出宝贵的意见与建议，相互分享与学习。
