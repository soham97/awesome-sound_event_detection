# Reading List for topics in Weakly Supervised Sound Event Detection

## Introduction
Sound event detection aims at processing the continuous acoustic signal and converting it into symbolic descriptions of the corresponding sound events present at the auditory scene. Sound event detection can be utilized in a variety of applications, including context-based indexing and retrieval in multimedia databases, unobtrusive monitoring in health care, and surveillance. Recently (since 2017), to utilise large multimedia data available, learning acoustic information from weak annotations was formulated. This reading list consists of papers which use weak annotation for learning symbolic descriptions of the corresponding sound events in the audio.

Papers covering multiple sub-areas are listed in both the sections. If there are any areas, papers, and datasets I missed, please let me know or feel free to make a pull request.

Maintained by [Soham Deshmukh](https://soham97.github.io)

## Table of Contents

* [Survey Papers](#survey-papers)
* [Areas](#areas)
  * [Learning formulation](#learning-formulation)
  * [Network architecture](#network-architecture)
  * [Pooling fuctions](#pooling-functions)
  * [Missing or noisy audio](#missing-or-noisy-audio)
  * [Generative Learning](#generative-learning)
  * [Representation Learning](#representation-learning)
  * [Multi-Task Learning](#multitask-learning)
  * [Adversarial Attacks](#adversarial-attacks)
  * [Few-Shot Learning](#few-shot-learning)
  * [Knowledge-transfer](#knowledge-transfer)
  * [Polyphonic SED](#polyphonic)
  * [Joint learning](#joint-task)
  * [Loss function](#loss)
  * [Audio and Visual](#audio-visual)
  * [Audio and Text [Audio Captioning]](#audio-text)
  * [Strongly and Weakly labelled data](#strong-weak)
  * [Healthcare](#healthcare)
  * [Robotics](#robotics)
* [Dataset](#dataset)
* [Workshops/Conferences/Journals](#WCJ)
* [Tutorials](#tutorials)
* [Courses](#courses)

# Research papers

## Survey papers
[Sound event detection and time–frequency segmentation from weakly labelled data](https://ieeexplore.ieee.org/abstract/document/8632940), TASLP 2019

## Areas

### Learning formulation
[Weakly supervised scalable audio content analysis](https://ieeexplore.ieee.org/abstract/document/7552989), ICME 2016

[Audio Event Detection using Weakly Labeled Data](https://dl.acm.org/doi/abs/10.1145/2964284.2964310), 24th ACM Multimedia Conference 2016

[An approach for self-training audio event detectors using web data](https://ieeexplore.ieee.org/abstract/document/8081532), 25th EUSIPCO 2017

[A joint detection-classification model for audio tagging of weakly labelled data](https://ieeexplore.ieee.org/abstract/document/7952234), ICASSP 2017

[Connectionist Temporal Localization for Sound Event Detection with Sequential Labeling](https://ieeexplore.ieee.org/abstract/document/8682278), ICASSP 2019

[Multi-Task Learning for Interpretable Weakly Labelled Sound Event Detection](https://arxiv.org/abs/2008.07085), ArXiv 2020

[A Sequential Self Teaching Approach for Improving Generalization in Sound Event Recognition](https://arxiv.org/abs/2007.00144), ICML 2020

[Non-Negative Matrix Factorization-Convolutional Neural Network (NMF-CNN) For Sound Event Detection](https://arxiv.org/abs/2001.07874), ArXiv 2020

[Duration robust weakly supervised sound event detection](https://arxiv.org/abs/1904.03841), ICASSP 2020

[SeCoST:: Sequential Co-Supervision for Large Scale Weakly Labeled Audio Event Detection](https://ieeexplore.ieee.org/abstract/document/9053613), ICASSP 2020

[Guided Learning for Weakly-Labeled Semi-Supervised Sound Event Detection](https://ieeexplore.ieee.org/abstract/document/9053584), ICASSP 2020

[Unsupervised Contrastive Learning of Sound Event Representations](https://arxiv.org/pdf/2011.07616.pdf), ICASSP 2021

[Sound Event Detection Based on Curriculum Learning Considering Learning Difficulty of Events](https://arxiv.org/abs/2102.05288), ICASSP 2021

### Network Architecture
[Weakly-supervised audio event detection using event-specific Gaussian filters and fully convolutional networks](https://ieeexplore.ieee.org/abstract/document/7952264), ICASSP 2017

[Deep CNN Framework for Audio Event Recognition using Weakly Labeled Web Data](https://arxiv.org/abs/1707.02530), NIPS Workshop on Machine Learning for Audio 2017

[Large-Scale Weakly Supervised Audio Classification Using Gated Convolutional Neural Network](https://ieeexplore.ieee.org/abstract/document/8461975), ICASSP 2018

[Orthogonality-Regularized Masked NMF for Learning on Weakly Labeled Audio Data](https://ieeexplore.ieee.org/abstract/document/8461293), ICASSP 2018

[Sound event detection and time–frequency segmentation from weakly labelled data](https://ieeexplore.ieee.org/abstract/document/8632940), TASLP 2019

[Attention-based Atrous Convolutional Neural Networks: Visualisation and Understanding Perspectives of Acoustic Scenes](https://ieeexplore.ieee.org/abstract/document/8683434), ICASSP 2019

[Sound Event Detection of Weakly Labelled Data With CNN-Transformer and Automatic Threshold Optimization](https://ieeexplore.ieee.org/abstract/document/9165887), TASLP 2020

[DD-CNN: Depthwise Disout Convolutional Neural Network for Low-complexity Acoustic Scene Classification](https://arxiv.org/abs/2007.12864), ArXiv 2020

[Effective Perturbation based Semi-Supervised Learning Method for Sound Event Detection](http://www.interspeech2020.org/index.php?m=content&c=index&a=show&catid=270&id=481), INTERSPEECH 2020

[Weakly-Supervised Sound Event Detection with Self-Attention](https://ieeexplore.ieee.org/abstract/document/9053609), ICASSP 2020

[Improving Deep Learning Sound Events Classifiers using Gram Matrix Feature-wise Correlations](https://arxiv.org/abs/2102.11771), ICASSP 2021

[An Improved Event-Independent Network for Polyphonic Sound Event Localization and Detection](https://arxiv.org/abs/2010.13092), ICASSP 2021

[AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778), ArXiv 2021

### Pooling functions
[Adaptive Pooling Operators for Weakly Labeled Sound Event Detection](https://dl.acm.org/doi/10.1109/TASLP.2018.2858559), TASLP 2018

[Comparing the Max and Noisy-Or Pooling Functions in Multiple Instance Learning for Weakly Supervised Sequence Learning Tasks](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/0990.pdf), Interspeech 2018

[A Comparison of Five Multiple Instance Learning Pooling Functions for Sound Event Detection with Weak Labeling](https://ieeexplore.ieee.org/abstract/document/8682847), ICASSP 2019

[Hierarchical Pooling Structure for Weakly Labeled Sound Event Detection](https://arxiv.org/pdf/1903.11791.pdf), INTERSPEECH 2019

[Weakly labelled audioset tagging with attention neural networks](https://ieeexplore.ieee.org/abstract/document/8777125), TASLP 2019

[Sound event detection and time–frequency segmentation from weakly labelled data](https://ieeexplore.ieee.org/abstract/document/8632940), TASLP 2019

[Multi-Task Learning for Interpretable Weakly Labelled Sound Event Detection](https://arxiv.org/abs/2008.07085), ArXiv 2019

### Missing or noisy audio:
[Sound event detection and time–frequency segmentation from weakly labelled data](https://ieeexplore.ieee.org/abstract/document/8632940), TASLP 2019

[Multi-Task Learning for Interpretable Weakly Labelled Sound Event Detection](https://arxiv.org/abs/2008.07085), ArXiv 2019

### Generative Learning
[Acoustic Scene Generation with Conditional Samplernn](https://ieeexplore.ieee.org/abstract/document/8683727), ICASSP 2019 

### Representation Learning
[Contrastive Predictive Coding of Audio with an Adversary](http://www.interspeech2020.org/index.php?m=content&c=index&a=show&catid=270&id=478), INTERSPEECH 2020

[ACCDOA: Activity-Coupled Cartesian Direction of Arrival Representation for Sound Event Localization and Detection](https://arxiv.org/abs/2010.15306), ICASSP 2021

### Multi-Task Learning
[Multi-Task Learning for Interpretable Weakly Labelled Sound Event Detection](https://arxiv.org/abs/2008.07085), ArXiv 2019

[Multi-Task Learning and post processing optimisation for sound event detection](http://dcase.community/documents/challenge2019/technical_reports/DCASE2019_Cances_69.pdf), DCASE 2019

[Label-efficient audio classification through multitask learning and self-supervision](https://arxiv.org/abs/1910.12587), ICLR 2019

### Few-Shot Learning

### Knowledge Transfer
[Transfer learning of weakly labelled audio](https://ieeexplore.ieee.org/abstract/document/8169984), WASPAA 2017

[Knowledge Transfer from Weakly Labeled Audio Using Convolutional Neural Network for Sound Events and Scenes](https://ieeexplore.ieee.org/abstract/document/8462200), ICASSP 2018

[PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://ieeexplore.ieee.org/abstract/document/9229505), TASLP 2020

### Polyphonic SED
[A first attempt at polyphonic sound event detection using connectionist temporal classification](https://ieeexplore.ieee.org/abstract/document/8682847), ICASSP 2017

[Polyphonic Sound Event Detection with Weak Labeling](http://www.cs.cmu.edu/~yunwang/papers/cmu-thesis.pdf), Thesis 2018

[Polyphonic Sound Event Detection and Localization using a Two-Stage Strategy](https://arxiv.org/abs/1905.00268), DCASE 2019

[Evaluation of Post-Processing Algorithms for Polyphonic Sound Event Detection](https://ieeexplore.ieee.org/abstract/document/8937143), WASPAA 2019

[Specialized Decision Surface and Disentangled Feature for Weakly-Supervised Polyphonic Sound Event Detection](https://ieeexplore.ieee.org/document/9076321), TASLP 2020

### Joint task
[A Joint Separation-Classification Model for Sound Event Detection of Weakly Labelled Data](https://ieeexplore.ieee.org/abstract/document/8462448), ICASSP 2018

[A Joint Framework for Audio Tagging and Weakly Supervised Acoustic Event Detection Using DenseNet with Global Average Pooling](http://www.interspeech2020.org/uploadfile/pdf/Mon-2-8-7.pdf), INTERSPEECH 2020

### Loss function
[Impact of Sound Duration and Inactive Frames on Sound Event Detection Performance](https://arxiv.org/abs/2102.01927v1), ICASSP 2021

### Audio and Visual
[A Light-Weight Multimodal Framework for Improved Environmental Audio Tagging](https://ieeexplore.ieee.org/abstract/document/8462479), ICASSP 2018

[Large Scale Audiovisual Learning of Sounds with Weakly Labeled Data](https://arxiv.org/abs/2006.01595), IJCAI 2020

### Audio and Text [Audio Captioning]
[Automated audio captioning with recurrent neural networks](https://ieeexplore.ieee.org/document/8170058), WASPAA 2017

[Audio caption: Listen and tell](https://arxiv.org/abs/1706.10006), ICASSP 2018

[AudioCaps: Generating captions for audios in the wild](https://www.aclweb.org/anthology/N19-1011/), NAACL 2019

[Audio Captioning Based on Combined Audio and Semantic Embeddings](https://ieeexplore.ieee.org/abstract/document/9327916), ISM 2020

[Clotho: An Audio Captioning Dataset](https://arxiv.org/abs/1910.09387), ICASSP 2020

[A Transformer-based Audio Captioning Model with Keyword Estimation](https://arxiv.org/abs/2007.00222), INTERSPEECH 2020

[Text-to-Audio Grounding: Building Correspondence Between Captions and Sound Events](https://arxiv.org/abs/2102.11474), ICASSP 2021

### Strongly and Weakly labelled data
[Audio event and scene recognition: A unified approach using strongly and weakly labeled data](https://ieeexplore.ieee.org/abstract/document/7966293), IJCNN 2017

### Others
[Sound Event Detection Using Point-Labeled Data](https://interactiveaudiolab.github.io/assets/papers/waspaa_2019_kim.pdf), WASPAA 2019

## Dataset
|Task |Dataset |Source |Num. Files |
| :--- | :--- | :--- | :--- |
|Sound Event Classification |[ESC-50](https://github.com/karolpiczak/ESC-50) |freesound.org |2k files |
|Sound Event Classification|[DCASE17 Task 4](http://dcase.community/challenge2017/task-large-scale-sound-event-detection)| YT videos|2k files |
|Sound Event Classification|[US8K](https://urbansounddataset.weebly.com/urbansound8k.html)| freesound.org|8k files |
|Sound Event Classification |[FSD50K](https://annotator.freesound.org/fsd/release/FSD50K/) |freesound.org |50k files |
|Sound Event Classification|[AudioSet](https://research.google.com/audioset/download.html)| YT videos|2M files |
|COVID-19 Detection using Coughs |[DiCOVA](https://dicova2021.github.io/) | Volunteers recording audio via a website  |  1k files|
|Few-shot Bioacoustic Event Detection|[DCASE21 Task 5](http://dcase.community/challenge2021/task-few-shot-bioacoustic-event-detection)| audio|4k+ files |
|Acoustic Scene Classification |[DCASE18 Task 1](http://dcase.community/challenge2019/task-acoustic-scene-classification#subtask-a)|Recorded by  TUT |1.5k |
|Various |[VGG-Sound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/)|Web videos |200k files|
|Audio Captioning   |[Clotho](https://github.com/audio-captioning/clotho-dataset) |freesound.org |5k files |
|Audio Captioning   |[AudioCaps](https://github.com/cdjkim/audiocaps/tree/master/dataset) |YT videos |51k files |
|Action Recognition |[UCF101](https://www.crcv.ucf.edu/research/data-sets/ucf101/) | Web videos  |  13k files|
|Unlabeled |[YFCC100M](https://pypi.org/project/yfcc100m/) | Yahoo videos  |  1M files|

[Other audio-based datasets to consider](https://homepages.tuni.fi/toni.heittola/datasets)

## Workshops/Conferences/Journals
List of old workshops (archived) and on-going workshops/conferences/journals:

[Machine Learning for Audio Signal Processing, NIPS 2017 workshop](https://nips.cc/Conferences/2017/Schedule?showEvent=8790)

[MLSP: Machine Learning for Signal Processing](https://ieeemlsp.cc/)

[WASPAA: IEEE Workshop on Applications of Signal Processing to Audio and Acoustics](https://www.waspaa.com)

[ICASSP: IEEE International Conference on Acoustics Speech and Signal Processing](https://2021.ieeeicassp.org/)

[INTERSPEECH](https://www.interspeech2021.org/)

[IEEE/ACM Transactions on Audio, Speech and Language Processing](https://dl.acm.org/journal/taslp)

[DCASE](http://dcase.community/)

## Tutorials

## Resources
[Computational Analysis of Sound Scenes and Events](https://www.springer.com/us/book/9783319634494)
