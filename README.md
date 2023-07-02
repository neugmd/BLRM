# Bayesian statistics-guided label refurbishment mechanism: Mitigating label noise in medical image classification
Pytorch implementation of Bayesian Statistics Guided Label Refurbishment Mechanism: Mitigating Label Noise in Medical Image Classification

[Mengdi Gao], [Ximeng Feng], [Mufeng Geng],[Zhe Jiang], [Lei Zhu], [Xiangxi Meng], [Chuanqing Zhou], [Qiushi Ren], [Yanye Lu]

[[`Paper`](https://doi.org/10.1002/mp.15799)]

> Purpose: Deep neural networks (DNNs) have been widely applied in medical image classification, benefiting from its powerful mapping capability among medical images.However, these existing deep learning-based methods depend on an enormous amount of carefully labeled images. Meanwhile, noise is inevitably introduced in the labeling process, degrading the performance of models. Hence, it is significant to devise robust training strategies to mitigate label noise in the medical image classification tasks.
> Methods: In this work, we propose a novel Bayesian statistics-guided label refurbishment mechanism (BLRM) for DNNs to prevent overfitting noisy images. BLRM utilizes maximum a posteriori probability in the Bayesian statistics and the exponentially time-weighted technique to selectively correct the labels of noisy images.The training images are purified gradually with the training epochs when BLRM is activated, further improving classification performance.
> Results: Comprehensive experiments on both synthetic noisy images (public OCT & Messidor datasets) and real-world noisy images (ANIMAL-10N) demonstrate that BLRM refurbishes the noisy labels selectively, curbing the adverse effects of noisy data. Also, the anti-noise BLRMs integrated with DNNs are effective at different noise ratio and are independent of backbone DNN architectures. In addition,BLRM is superior to state-of -the-art comparative methods of anti-noise.
> Conclusions: These investigations indicate that the proposed BLRM is well capable of mitigating label noise in medical image classification tasks.

## Installation

Install the dependencies listed in the `requirements.txt` file.

```
pip install -r requirements.txt
```

## Citing this paper

If you use BLRM in your research, please use the following BibTeX entry.

```
@article{gao2022bayesian,
  title={Bayesian statistics-guided label refurbishment mechanism: Mitigating label noise in medical image classification},
  author={Gao, Mengdi and Feng, Ximeng and Geng, Mufeng and Jiang, Zhe and Zhu, Lei and Meng, Xiangxi and Zhou, Chuanqing and Ren, Qiushi and Lu, Yanye},
  journal={Medical Physics},
  volume={49},
  number={9},
  pages={5899--5913},
  year={2022},
  publisher={Wiley Online Library}
}
```
