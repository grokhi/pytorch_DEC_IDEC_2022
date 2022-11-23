# PyTorch DEC/IDEC (2022)
Rewritten Deep Embedded Clustering (DEC) and Improved DEC (IDEC) algorithms for the current version of pytorch. Obtained code was implemented then for the purposes of text feature extraction. (See [this paper](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-022-00564-9#:~:text=Based%20on%20the%20results%2C%20BERT,that%20positions%20similar%20texts%20closer.) for details)

Original papers: 
- DEC [Unsupervised Deep Embedding for Clustering Analysis (2016) by Junyuan Xie et al.](https://arxiv.org/abs/1511.06335) 
- IDEC [Improved Deep Embedded Clustering with Local Structure Preservation (2017) by Xifeng Guo et al.](https://www.researchgate.net/publication/317095655_Improved_Deep_Embedded_Clustering_with_Local_Structure_Preservation)

# Results
Perfomance of the algortihms was tested on the MNIST. Cluster accuracy achieves around 85% (Hungarian algorithm, see [scipy desc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)) and depends to accuracy of initially generated centroids (kmeans on the encoder outputs)
## DEC
confm1

## IDEC
confm2

# Code references
- [Original DEC (caffe)](https://github.com/piiswrong/dec)
- [Original DEC/IDEC (keras)](https://github.com/XifengGuo/IDEC)
- [pt-dec (pytroch)](https://github.com/vlukiyanov/pt-dec) 
- [Another DEC (pytorch-lightning)](https://github.com/youngerous/dec-pytorch)

# TODO
* Dockerize
* docs
* tuning
* .py files
