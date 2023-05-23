# MGR: Multi-generator based Rationalization   
This repo contains Pytorch implementation of [MGR: Multi-generator based Rationalization  (ACL 2023 main conference)](https://arxiv.org/abs/2305.04492).    

Preparing code can be an exhausting task. Please consider giving our repository a star as a token of encouragement.  If you have any questions, just open an issue or send us an e-mail.      
Thank you! 


## Environments
RTX3090

Create an environment with: conda create -n MGR python=3.7.13

Then activate it: conda activate MGR

Install pytorch: conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

Install other packages: pip install -r requirements.txt


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

The last line "The annotation performance: sparsity: 20.2753, precision: 76.3473, recall: 83.6095, f1: 79.8135 " indicates the overlap between the selected tokens and human-annotated rationales. The penultimate line shows the predictive accuracy on the test set. 


 


