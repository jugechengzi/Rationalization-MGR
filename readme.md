There was a minor issue in the previous version of our paper. Specifically, Equation 4 should read as follows: $\mathop{\min}\limits_{\theta_{G_1},\ldots,\theta_{G_n}, \theta_P}\sum\limits_{(X,Y)\sim \mathcal{D}} H(Y,\hat{Y})=\sum\limits_{(X,Y)\sim \mathcal{D}} \sum\limits_{i=1}^{n} H(Y,f_{P}(f_{G_i}(X)))$.  
We have rectified this error in our [arxiv](https://arxiv.org/abs/2305.04492) version.

# MGR: Multi-generator based Rationalization   
This repo contains Pytorch implementation of [MGR: Multi-generator based Rationalization  (ACL 2023 oral)](https://github.com/jugechengzi/Rationalization-MGR/blob/main/acl2023.pdf).  

You can also refer to our team's other complementary work in this series：[FR (NeurIPS2022)](https://arxiv.org/abs/2209.08285), [DR (KDD 2023)](https://dl.acm.org/doi/abs/10.1145/3580305.3599299), [MGR (ACL 2023)](https://arxiv.org/abs/2305.04492), [MCD (NeurIPS 2023)](https://arxiv.org/abs/2309.13391), [DAR (ICDE 2024)](https://arxiv.org/abs/2312.04103).

**If the code has any bugs, please open an issue. We will be grateful for your help.**

**We have provided some tips for you to better understand our code and build your method on top of it: [tips](https://github.com/jugechengzi/Rationalization-MGR/blob/main/tips.pdf).** If you're still having trouble reproducing your code after reading these tips, I'd be happy to set up a video meeting to help you.

I am happy to say that a recent paper called [SSR](https://arxiv.org/abs/2403.07955) has taken our previous work [FR](https://github.com/jugechengzi/FR) as a backbone. Congratulations on the great work of Yue et.al. and thanks for their citation. I also find that several recent works, such as [GR](https://ojs.aaai.org/index.php/AAAI/article/download/29783/31352) and [YOFO](https://arxiv.org/abs/2311.02344), have designed their experiments based on our open source code. Congratulations to all of them and I am really happy to know that my work is helping others.



## Environments
RTX3090

Create an environment with: conda create -n MGR python=3.7.13

Then activate it: conda activate MGR

Install pytorch: conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

Install other packages: pip install -r requirements.txt


Due to different versions of torch, you may need to replace "cls_loss = args.cls_lambda * F.cross_entropy(forward_logit, labels)" with "cls_loss = args.cls_lambda * F.cross_entropy(forward_logit, labels.long())"



## Datasets  
For Beer Reviews, you should first obtain authorization for this dataset from the original author.
 
Beer Reviews: you can get it [here](http://people.csail.mit.edu/taolei/beer/). Then place it in the ./data/beer directory.  
Hotel Reviews: you can get it [here](https://people.csail.mit.edu/yujia/files/r2a/data.zip). 
Then  find hotel_Location.train, hotel_Location.dev, hotel_Service.train, hotel_Service.dev, hotel_Cleanliness.train, hotel_Cleanliness.dev from data/oracle and put them in the ./data/hotel directory. 
Find hotel_Location.train, hotel_Service.train, hotel_Cleanliness.train from data/target and put them in the ./data/hotel/annotations directory.  
Word embedding: [glove.6B.100d.txt](https://nlp.stanford.edu/projects/glove/). Then put it in the ./data/hotel/embeddings directory.


## Running Example  
### Correlated Beer
#### Appearance:  
For sparsity $S\approx 10$:   
python -u norm_beer.py --correlated 1 --lr 0.00003 --batch_size 128 --gpu 0 --sparsity_percentage 0.083 --epochs 400 --aspect 0

For sparsity $S\approx 20$:   
python -u norm_beer.py --correlated 1 --lr 0.00003 --batch_size 128 --gpu 0 --sparsity_percentage 0.173 --epochs 400 --aspect 0

For sparsity $S\approx 30$:   
python -u norm_beer.py --correlated 1 --lr 0.00003 --batch_size 128 --gpu 0 --sparsity_percentage 0.273 --epochs 400 --aspect 0 

### Hotel  
cleanliness:  
python -u norm_beer.py --data_type hotel --lr 0.00007 --batch_size 1024 --gpu 0 --sparsity_percentage 0.1 --sparsity_lambda 10 --continuity_lambda 10 --epochs 800 --aspect 2


**_Notes_**: "--sparsity_percentage 0.173" means "$s=0.173$" in Eq.3 (But the actual sparsity is different from $s$. See the results below.). "--sparsity_lambda 10 --continuity_lambda 10 " means $\lambda_1=10, \lambda_2=10$. "--epochs 400" means we run 400 epochs and take the results when the "dev_acc" is best.  

When you change the random seed, you need to adjust the "sparsity_percentage" according to the actual sparsity on the test set.



## Result
You will get a result like "best_dev_epoch=194" at last. Then you need to find the result corresponding to the epoch with number "194".  
For Beer-Appearance, you may get a result like:  

Train time for epoch #194 : 23.373237 second  
traning epoch:194 recall:0.7490 precision:0.8077 f1-score:0.7772 accuracy:0.7853  
Validate  
dev epoch:194 recall:0.7430 precision:0.7528 f1-score:0.7479 accuracy:0.7495  
Validate Sentence  
dev dataset : recall:0.6829 precision:0.7212 f1-score:0.7015 accuracy:0.7094  
Annotation  
annotation dataset : recall:0.8505 precision:0.9987 f1-score:0.9187 accuracy:0.8515  
The annotation performance: sparsity: 20.2753, precision: 76.3473, recall: 83.6095, f1: 79.8135  
Annotation Sentence  
annotation dataset : recall:0.8548 precision:0.9987 f1-score:0.9212 accuracy:0.8558  
Rationale  
rationale dataset : recall:0.8375 precision:0.9974 f1-score:0.9105 accuracy:0.8376  

The last line "The annotation performance: sparsity: 20.2753, precision: 76.3473, recall: 83.6095, f1: 79.8135 " indicates the overlap between the selected tokens and human-annotated rationales. 


 ## Citation   
 @inproceedings{liu-etal-2023-mgr,  
    title = "{MGR}: Multi-generator Based Rationalization",  
    author = "Liu, Wei  and  
      Wang, Haozhao  and  
      Wang, Jun  and  
      Li, Ruixuan  and  
      Li, Xinyang  and  
      Zhang, YuanKai  and  
      Qiu, Yang",  
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",  
    month = july,  
    year = "2023",  
    address = "Toronto, Canada",  
    publisher = "Association for Computational Linguistics",  
    url = "https://aclanthology.org/2023.acl-long.715",  
    pages = "12771--12787",  
}  


## Acknowledgement

The code is largely based on [Car](https://github.com/code-terminator/classwise_rationale) and [DMR](https://github.com/kochsnow/distribution-matching-rationality). Most of the hyperparameters (e.g. the '--cls_lambda'=0.9) are also from them. We are grateful for their open source code.

