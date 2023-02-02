# Reading List for topics in Sound Event Detection
## Introduction
Sound event detection aims at processing the continuous acoustic signal and converting it into symbolic descriptions of the corresponding sound events present at the auditory scene. Sound event detection can be utilized in a variety of applications, including context-based indexing and retrieval in multimedia databases, unobtrusive monitoring in health care, and surveillance. Since 2017, to utilise large multimedia data available, learning acoustic information from weak annotations was formulated. This reading list consists of papers for sound event detection and Sound AI.

Papers covering multiple sub-areas are listed in both the sections. If there are any areas, papers, and datasets I missed, please let me know or feel free to make a pull request.

### The reading list is expanded to include topics in Sound AI

Maintained by [Soham Deshmukh](https://soham97.github.io)

## Recent Content
INTERSPEECH 2022 papers added <br>
ICASSP 2022 papers added <br>
WASPAA 2021 papers added <br>
INTERSPEECH 2021 papers added <br>
ICASSP 2021 papers added

## Table of Contents

* [Survey Papers](#survey-papers)
* [Areas](#areas)
  * [Learning formulation](#learning-formulation)
  * [Network architecture](#network-architecture)
  * [Pooling fuctions](#pooling-functions)
  * [Missing or noisy audio](#missing-or-noisy-audio)
  * [Data Augmentation](#data-augmentation)
  * [Audio Generation](#audio-generation)
  * [Representation Learning](#representation-learning)
  * [Multi-Task Learning](#multi-task-learning)
  * [Adversarial Attacks](#adversarial-attacks)
  * [Few-Shot](#few-shot)
  * [Zero-Shot](#zero-shot)
  * [Knowledge-transfer](#knowledge-transfer)
  * [Polyphonic SED](#polyphonic-sed)
  * [Loss function](#loss-function)
  * [Audio and Visual](#audio-and-visual)
  * [Audio Captioning](#audio-captioning)
  * [Audio Retrieval](#audio-retrieval)
  * [Healthcare](#healthcare)
  * [Robotics](#robotics)
* [Dataset](#dataset)
* [Workshops/Conferences/Journals](#workshops-conferences-journals)
* [More](#more)

# Research papers

## Survey papers
[Sound event detection and time–frequency segmentation from weakly labelled data](https://ieeexplore.ieee.org/abstract/document/8632940), TASLP 2019

[Sound Event Detection: A Tutorial](https://arxiv.org/abs/2107.05463), ArXiv 2021

[Automated Audio Captioning: an Overview of Recent Progress and New Challenges](), ArXiv 2022

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

[Comparison of Deep Co-Training and Mean-Teacher Approaches for Semi-Supervised Audio Tagging](https://ieeexplore.ieee.org/document/9415116), ICASSP 2021

[Enhancing Audio Augmentation Methods with Consistency Learning](https://ieeexplore.ieee.org/document/9414316), ICASSP 2021

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

[AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778), INTERSPEECH 2021

[Event Specific Attention for Polyphonic Sound Event Detection](https://assets.amazon.science/31/fc/503f716a43fda8e0718e7ffc6e9e/event-specific-attention-for-polyphonic-sound-event-detection.pdf), INTERSPEECH 2021

[Sound Event Detection with Adaptive Frequency Selection](https://arxiv.org/abs/2105.07596), WASPAA 2021

[SSAST: Self-Supervised Audio Spectrogram Transformer](https://ojs.aaai.org/index.php/AAAI/article/view/21315), AAAI 2022

[HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection](https://arxiv.org/abs/2202.00874), ICASSP 2022

[MAE-AST: Masked Autoencoding Audio Spectrogram Transformer](https://arxiv.org/abs/2203.16691), INTERSPEECH 2022

[Efficient Training of Audio Transformers with Patchout](https://arxiv.org/abs/2110.05069), INTERSPEECH 2022

[BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/pdf/2212.09058.pdf), ArXiv 2022

### Pooling functions
[Adaptive Pooling Operators for Weakly Labeled Sound Event Detection](https://dl.acm.org/doi/10.1109/TASLP.2018.2858559), TASLP 2018

[Comparing the Max and Noisy-Or Pooling Functions in Multiple Instance Learning for Weakly Supervised Sequence Learning Tasks](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/0990.pdf), Interspeech 2018

[A Comparison of Five Multiple Instance Learning Pooling Functions for Sound Event Detection with Weak Labeling](https://ieeexplore.ieee.org/abstract/document/8682847), ICASSP 2019

[Hierarchical Pooling Structure for Weakly Labeled Sound Event Detection](https://arxiv.org/pdf/1903.11791.pdf), INTERSPEECH 2019

[Weakly labelled audioset tagging with attention neural networks](https://ieeexplore.ieee.org/abstract/document/8777125), TASLP 2019

[Sound event detection and time–frequency segmentation from weakly labelled data](https://ieeexplore.ieee.org/abstract/document/8632940), TASLP 2019

[Multi-Task Learning for Interpretable Weakly Labelled Sound Event Detection](https://arxiv.org/abs/2008.07085), ArXiv 2019

[A Global-Local Attention Framework for Weakly Labelled Audio Tagging](https://ieeexplore.ieee.org/document/9414357), ICASSP 2021

### Missing or noisy audio:
[Sound event detection and time–frequency segmentation from weakly labelled data](https://ieeexplore.ieee.org/abstract/document/8632940), TASLP 2019

[Multi-Task Learning for Interpretable Weakly Labelled Sound Event Detection](https://arxiv.org/abs/2008.07085), ArXiv 2019

[Improving weakly supervised sound event detection with self-supervised auxiliary tasks](https://arxiv.org/abs/2106.06858), INTERSPEECH 2021

### Data Augmentation:
[SpecAugment++: A Hidden Space Data Augmentation Method for Acoustic Scene Classification](https://arxiv.org/abs/2103.16858), INTERSPEECH 2021

### Representation Learning
[Contrastive Predictive Coding of Audio with an Adversary](http://www.interspeech2020.org/index.php?m=content&c=index&a=show&catid=270&id=478), INTERSPEECH 2020

[Towards Learning a Universal Non-Semantic Representation of Speech](https://arxiv.org/abs/2002.12764), INTERSPEECH 2021

[ACCDOA: Activity-Coupled Cartesian Direction of Arrival Representation for Sound Event Localization and Detection](https://arxiv.org/abs/2010.15306), ICASSP 2021

[FRILL: A Non-Semantic Speech Embedding for Mobile Devices](https://arxiv.org/abs/2011.04609), INTERSPEECH 2021

[HEAR 2021: Holistic Evaluation of Audio Representations](https://arxiv.org/abs/2203.03022), PMLR: NeurIPS 2021 Competition Track

[Conformer-Based Self-Supervised Learning for Non-Speech Audio Tasks](https://arxiv.org/abs/2110.07313), ICASSP 2022

[Towards Learning Universal Audio Representations](https://arxiv.org/pdf/2111.12124.pdf), ICASSP 2022

[SSAST: Self-Supervised Audio Spectrogram Transformer](https://ojs.aaai.org/index.php/AAAI/article/view/21315), AAAI 2022

### Multi-Task Learning
[A Joint Separation-Classification Model for Sound Event Detection of Weakly Labelled Data](https://ieeexplore.ieee.org/abstract/document/8462448), ICASSP 2018

[Multi-Task Learning for Interpretable Weakly Labelled Sound Event Detection](https://arxiv.org/abs/2008.07085), ArXiv 2019

[Multi-Task Learning and post processing optimisation for sound event detection](http://dcase.community/documents/challenge2019/technical_reports/DCASE2019_Cances_69.pdf), DCASE 2019

[Label-efficient audio classification through multitask learning and self-supervision](https://arxiv.org/abs/1910.12587), ICLR 2019

[A Joint Framework for Audio Tagging and Weakly Supervised Acoustic Event Detection Using DenseNet with Global Average Pooling](http://www.interspeech2020.org/uploadfile/pdf/Mon-2-8-7.pdf), INTERSPEECH 2020

[Improving weakly supervised sound event detection with self-supervised auxiliary tasks](https://arxiv.org/abs/2106.06858), INTERSPEECH 2021

[Identifying Actions for Sound Event Classification](https://arxiv.org/abs/2104.12693), WASPAA 2021

[Impact of Acoustic Event Tagging on Scene Classification in a Multi-Task Learning Framework](https://arxiv.org/abs/2206.13476), INTERSPEECH 2022

### Few-Shot
[Few-Shot Audio Classification with Attentional Graph Neural Networks](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1532.pdf), INTERSPEECH 2019

[Continual Learning of New Sound Classes Using Generative Replay](https://ieeexplore.ieee.org/document/8937236), WASSPA 2019

[Few-Shot Sound Event Detection](https://ieeexplore.ieee.org/document/9054708), ICASSP 2020

[Few-Shot Continual Learning for Audio Classification](https://ieeexplore.ieee.org/document/9413584), ICASSP 2021

[Unsupervised and Semi-Supervised Few-Shot Acoustic Event Classification](https://ieeexplore.ieee.org/document/9414546), ICASSP 2021

[Who Calls the Shots? Rethinking Few-Shot Learning for Audio](https://arxiv.org/abs/2110.09600), WASPAA 2021

[A Mutual Learning Framework For Few-Shot Sound Event Detection](https://arxiv.org/abs/2110.04474), ICASSP 2022

[Active Few-Shot Learning for Sound Event Detection](https://www.isca-speech.org/archive/pdfs/interspeech_2022/wang22aa_interspeech.pdf), INTERSPEECH 2022

### Zero-Shot
[AudioCLIP: Extending CLIP to Image, Text and Audio](https://arxiv.org/abs/2106.13043), ICASSP 2022

[Wav2CLIP: Learning Robust Audio Representations From CLIP](https://arxiv.org/abs/2111.12124), ICASSP 2022

[CLAP 👏: Learning Audio Concepts From Natural Language Supervision](https://arxiv.org/abs/2206.04769), ArXiv 2022

[Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation](https://arxiv.org/abs/2211.06687), ArXiv 2022

### Knowledge Transfer
[Transfer learning of weakly labelled audio](https://ieeexplore.ieee.org/abstract/document/8169984), WASPAA 2017

[Knowledge Transfer from Weakly Labeled Audio Using Convolutional Neural Network for Sound Events and Scenes](https://ieeexplore.ieee.org/abstract/document/8462200), ICASSP 2018

[PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://ieeexplore.ieee.org/abstract/document/9229505), TASLP 2020

[Do sound event representations generalize to other audio tasks? A case study in audio transfer learning](https://arxiv.org/abs/2106.11335), INTERSPEECH 2021

### Polyphonic SED
[A first attempt at polyphonic sound event detection using connectionist temporal classification](https://ieeexplore.ieee.org/abstract/document/8682847), ICASSP 2017

[Polyphonic Sound Event Detection with Weak Labeling](http://www.cs.cmu.edu/~yunwang/papers/cmu-thesis.pdf), Thesis 2018

[Polyphonic Sound Event Detection and Localization using a Two-Stage Strategy](https://arxiv.org/abs/1905.00268), DCASE 2019

[Evaluation of Post-Processing Algorithms for Polyphonic Sound Event Detection](https://ieeexplore.ieee.org/abstract/document/8937143), WASPAA 2019

[Specialized Decision Surface and Disentangled Feature for Weakly-Supervised Polyphonic Sound Event Detection](https://ieeexplore.ieee.org/document/9076321), TASLP 2020

[Spatial Data Augmentation with Simulated Room Impulse Responses for Sound Event Localization and Detection](https://arxiv.org/abs/2111.12124), ICASSP 2022

### Loss function
[Impact of Sound Duration and Inactive Frames on Sound Event Detection Performance](https://arxiv.org/abs/2102.01927v1), ICASSP 2021

### Audio and Visual
[A Light-Weight Multimodal Framework for Improved Environmental Audio Tagging](https://ieeexplore.ieee.org/abstract/document/8462479), ICASSP 2018

[Large Scale Audiovisual Learning of Sounds with Weakly Labeled Data](https://arxiv.org/abs/2006.01595), IJCAI 2020

[Labelling unlabelled videos from scratch with multi-modal self-supervision](https://arxiv.org/abs/2006.13662), NeurIPS 2020

[Audio-Visual Event Recognition Through the Lens of Adversary](https://ieeexplore.ieee.org/document/9415065), ICASSP 2021

[Taming Visually Guided Sound Generation](https://arxiv.org/abs/2110.08791), BMVC 2021

### Audio Captioning
[Automated audio captioning with recurrent neural networks](https://ieeexplore.ieee.org/document/8170058), WASPAA 2017

[Audio caption: Listen and tell](https://arxiv.org/abs/1706.10006), ICASSP 2018

[AudioCaps: Generating captions for audios in the wild](https://www.aclweb.org/anthology/N19-1011/), NAACL 2019

[Audio Captioning Based on Combined Audio and Semantic Embeddings](https://ieeexplore.ieee.org/abstract/document/9327916), ISM 2020

[Clotho: An Audio Captioning Dataset](https://arxiv.org/abs/1910.09387), ICASSP 2020

[A Transformer-based Audio Captioning Model with Keyword Estimation](https://arxiv.org/abs/2007.00222), INTERSPEECH 2020

[Text-to-Audio Grounding: Building Correspondence Between Captions and Sound Events](https://arxiv.org/abs/2102.11474), ICASSP 2021

[Learning Contextual Tag Embeddings for Cross-Modal Alignment of Audio and Tags](https://ieeexplore.ieee.org/document/9414638), ICASSP 2021

[Automated Audio Captioning using Transfer Learning and Reconstruction Latent Space Similarity Regularization](https://arxiv.org/abs/2108.04692), ICASSP 2022

[Sound Event Detection Guided by Semantic Contexts of Scenes](https://arxiv.org/abs/2110.03243), ICASSP 2022

[Interactive Audio-text Representation for Automated Audio Captioning with Contrastive Learning](https://arxiv.org/abs/2203.15526), INTERSPEECH 2022

### Audio Retrieval
[Audio Retrieval with Natural Language Queries: A Benchmark Study](https://ieeexplore.ieee.org/abstract/document/9707629), IEEE Transactions on Multimedia 2022

[On Metric Learning for Audio-Text Cross-Modal Retrieval](https://arxiv.org/abs/2203.15537), INTERSPEECH 2022

[Introducing Auxiliary Text Query-modifier to Content-based Audio Retrieval](https://arxiv.org/abs/2207.09732), INTERSPEECH 2022

[Audio Retrieval with WavText5K and CLAP Training](https://arxiv.org/abs/2209.14275), ArXiv 2022

### Audio Generation
[Acoustic Scene Generation with Conditional Samplernn](https://ieeexplore.ieee.org/abstract/document/8683727), ICASSP 2019

[Conditional Sound Generation Using Neural Discrete Time-Frequency Representation Learning](https://ieeexplore.ieee.org/abstract/document/9596430), MLSP 2021

[Taming Visually Guided Sound Generation](https://arxiv.org/abs/2110.08791), BMVC 2021

[Diffsound: Discrete Diffusion Model for Text-to-sound Generation](https://arxiv.org/abs/2207.09983), ArXiv 2022

[AudioGen: Textually Guided Audio Generation](https://felixkreuk.github.io/text2audio_arxiv_samples/paper.pdf), ICML 2023

[Make-An-Audio: Text-To-Audio Generation with Prompt-Enhanced Diffusion Models](https://text-to-audio.github.io/paper.pdf), 2023

[AudioLDM: Text-to-Audio Generation with Latent Diffusion Models](https://arxiv.org/abs/2301.12503), ArXiv 2023

### Others
[Audio event and scene recognition: A unified approach using strongly and weakly labeled data](https://ieeexplore.ieee.org/abstract/document/7966293), IJCNN 2017

[Sound Event Detection Using Point-Labeled Data](https://interactiveaudiolab.github.io/assets/papers/waspaa_2019_kim.pdf), WASPAA 2019

[An Investigation of the Effectiveness of Phase for Audio Classification](https://arxiv.org/abs/2110.02878), ICASSP 2022

# Dataset
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
|Audio Retrieval    |[SoundDescs](https://github.com/akoepke/audio-retrieval-benchmark) | BBC Sound Effects | 32k files |
|Audio Retrieval    |[WavText5K](https://github.com/microsoft/WavText5K) | Varied | 5k files |
|Action Recognition |[UCF101](https://www.crcv.ucf.edu/research/data-sets/ucf101/) | Web videos  |  13k files|
|Unlabeled |[YFCC100M](https://pypi.org/project/yfcc100m/) | Yahoo videos  |  1M files|

[Other audio-based datasets to consider](https://homepages.tuni.fi/toni.heittola/datasets) <br>
[DCASE dataset list](https://dcase-repo.github.io/dcase_datalist/)

# Workshops/Conferences/Journals
List of old workshops (archived) and on-going workshops/conferences/journals:
|Venues |link |
| :--- | :--- |
|Machine Learning for Audio Signal Processing, NIPS 2017 workshop| https://nips.cc/Conferences/2017/Schedule?showEvent=8790 |
|MLSP: Machine Learning for Signal Processing| https://ieeemlsp.cc/ |
|WASPAA: IEEE Workshop on Applications of Signal Processing to Audio and Acoustics| https://www.waspaa.com |
|ICASSP: IEEE International Conference on Acoustics Speech and Signal Processing| https://2021.ieeeicassp.org/ |
|INTERSPEECH| https://www.interspeech2021.org/ |
|IEEE/ACM Transactions on Audio, Speech and Language Processing| https://dl.acm.org/journal/taslp |
|DCASE| http://dcase.community/ |

# Resources
[Computational Analysis of Sound Scenes and Events](https://www.springer.com/us/book/9783319634494)

# More
- If you are interested in audio-captioning, [K. Drossos](https://github.com/dr-costas) maintains a detailed reading list [here](https://github.com/audio-captioning/audio-captioning-papers)
- Tracking states of the arts and recent results (bibliography) on sound AI topics and audio tasks [here](https://github.com/soham97/sound_ai_progress)
