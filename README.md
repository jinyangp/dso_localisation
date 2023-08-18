<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<div align="center">
  <h2>Deep ML Wireless Localisation using Donor Signals with Deep Learning</h2>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#folder-description">Folder Description</a></li>
        <li><a href="#results">Results</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

![Project Description Image](https://github.com/jinyangp/dso_localisation/assets/85600715/458a60c2-432d-4c4e-a4de-297360400037)

Traditional localisation relies on GNSS to perform accurate localisation. However, this requires a few Line-of-Sight (LOS) transmission (at least three) between the mobile node (receiver) and the base stations. However, in urban environments, there may not be an availability of LOS transmission due to physical blockages like high-rise buildings. This can lead to a huge deterioration in the performance of traditional methods using GNSS.

Hence, this project aims to develop other means of performing localisation in urban environments by leveraging on Non-Line-of-Sight (NLOS) transmissions from donor signals.

In urban environments, donor signals are frequently transmitted for communications and broadcasting purposes over base stations. Using a suitable receiver, the receiver is able to receive the transmitted donor signals from these base stations. These transmissions are in the form of LOS and NLOS, but mainly in NLOS. NLOS transmissions arise from signals reflecting off surfaces, such as high-rise buildings, in the environment.

Using ray-tracing softwares, NLOS transmissions arriving from multiple paths can be captured and these NLOS transmissions can be used to supplement the lack of LOS transmissions needed for accurate localisation.

In order to utilise these NLOS transmissions to achieve accurate localisation, fingerprinting techniques will be used in conjunction with Deep Learning.  

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Folder Description

This section gives a brief description of the content inside each folder.

**Note: Some notebooks use anchor tag to denote different sections of the notebook. These anchor tags do not work when the notebook is viewed on GitHub. Please head to [nbviewer](https://nbviewer.org/) and copy the URL of the notebook to view the notebook with functional anchor tags.**

| Folder name | Description |
| ----------- | ----------- |
| data_augmentation | Contains implementation code for performing data augmentation using the GraphSAGE model and non machine learning methods |
| models_evaluation | Contains implementation code for visualising the cumulative distribution function (CDF) of distance error for each of the model developed |
| mpri | Contains implementation code for developing, tuning and experimentations performed using the MPRI model |
| resnet18 | Contains implementation code for developing, tuning and experimentations performed using the ResNet-18 model |
| wknn | Contains implementation code for developing, tuning and experimentations performed using the Locality-Sensitve Hashing Weighted K-Nearest Neighbour (LSH-WKNN) model |
| xception | Contains implementation code for developing, tuning and experimentations performed using the Xception model |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Results

The raw and augmented datasets along with the trained weights of models on selected datasets can be found in the link below.
A more comprehensive documentation of the provided datasets and model weights is included in the link provided below.

[Datasets and Weights](https://drive.google.com/file/d/1rf1ddxMqUGHHmMdavZGxHp8ri43ERqgR/view?usp=sharing)

**Note: The files were uploaded as a .zip folder.**

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

The deep learning models in this project were created using Tensorflow and Keras.

[![Python][Python-img]][Python-url] [![Tensorflow][Tensorflow-img]][Tensorflow-url] [![Keras][Keras-img]][Keras-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these steps.

### Prerequisites

Python and pip is needed before proceeding further. Python can be installed [here](Python-url).

1. pip
```sh
python get-pip.py
```

### Installation

1. Clone the repo
```sh
git clone https://github.com/jinyangp/dso_localisation.git
```

2. Install virtualenv package
```sh
pip install virtualenv
```

3. Create a virtual environment in desired directory
```sh
cd [project path]
virtualenv venv
```

4. Activate the environment
```sh
source ./venv/bin/activate
```

5. Install requirements packages
```sh
pip install -r [requirements file]
```
   
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

The following past works were referenced:

1. MPRI Model: K. Gao, H. Wang, H. Lv and W. Liu, "Toward 5G NR High-Precision Indoor Positioning via Channel Frequency Response: A New Paradigm and Dataset Generation Method," in IEEE Journal on Selected Areas in Communications, vol. 40, no. 7, pp. 2233-2247, July 2022, doi: 10.1109/JSAC.2022.3157397.

2. ResNet18 Model: Wu Jimmy (2021). resnet18-tf2. https://github.com/jimmyyhwu/resnet18-tf2

3. Xception Model: Chollet, F. (2016). Xception: Deep Learning with Depthwise Separable Convolutions. arXiv (Cornell University). https://doi.org/10.48550/arxiv.1610.02357
   
4. LSH-WKNN: X. Wang, L. Liu, Y. Lin and X. Chen, "A Fast Single-Site Fingerprint Localization Method in Massive MIMO System," 2019 11th International Conference on Wireless Communications and Signal Processing (WCSP), Xi'an, China, 2019, pp. 1-6, doi: 10.1109/WCSP.2019.8927853.

5. Data Augmentation using Graph Signal Processing: Y. Chen, G. Li, Y. Tan and G. Zhang, "Graph-Based Radio Fingerprint Augmentation for Deep-Learning-Based Indoor Localization," in IEEE Sensors Journal, vol. 23, no. 6, pp. 6074-6084, 15 March15, 2023, doi:10.1109/JSEN.2023.3242641.
   
6. GraphSAGE model: Hamilton, W. L., Ying, R., & Leskovec, J. (2017b). Inductive representation learning on large graphs. arXiv (Cornell University). https://doi.org/10.48550/arxiv.1706.02216


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[Python-img]: https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[Tensorflow-img]: https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white
[Tensorflow-url]: https://www.tensorflow.org/
[Keras-img]: https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white
[Keras-url]: https://keras.io/
