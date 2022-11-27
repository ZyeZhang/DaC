# Official implementation for [**DaC (NeurIPS 2022)**](https://arxiv.org/abs/2211.06612)

Code for our NeurIPS 2022 paper 'Divide and Contrast: Source-free Domain Adaptation via Adaptive Contrastive Learning.' [Paper (openreview)](https://openreview.net/forum?id=NjImFaBEHl)

### Dataset:
Please manually download the datasets [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view), [VisDA-C](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification), [DomainNet](http://ai.bu.edu/DomainNet/) from the official websites, and denote the path of images in each '.txt' (best under folder /DATANAME/data/).

### VisDA Training:
```python
 cd VisDA/
 sh run_source
 sh run_train
```
  
### Citation

We would be grateful if you cite this paper:

```
@inproceedings{zhang2022dac, 
 title={Divide and Contrast: Source-free Domain Adaptation via Adaptive Contrastive Learning.}, 
 author={Zhang, Ziyi and Chen, Weikai and Cheng, Hui and Li, Zhen and Li, Siyuan and Lin, Liang and Li, Guanbin}, 
 booktitle={Conference on Neural Information Processing Systems (NeurIPS)},  
 year={2022}
}

```


### Contact

- [zhangziyi@lamda.nju.edu.cn](zhangziyi@lamda.nju.edu.cn)

### Acknowledgement
This code is based on [SHOT](https://github.com/tim-learn/SHOT)
