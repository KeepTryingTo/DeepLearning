# RATLIP
> Paper: [![arXiv](https://img.shields.io/badge/arXiv-2405.00354-b31b1b.svg)](https://arxiv.org/abs/2405.08114)

RATLIP: Generative Adversarial CLIP Text-to-Image Synthesis Based  on Recurrent Affine Transformations
![](./rat.png)


## Requirements

**At least 1x24GB 3090 GPU (for training), only CPU (for sampling)**

1. **Environment**

```Bash
conda create -n RATLIP python=3.9
conda activate RATLIP

```

2. **Clone this repo**

```Bash
git clone https://github.com/OxygenLu/RATLIP.git
```

3. **Install the requirements**

```Bash
cd RATLIP
pip install -r requirements.txt

```

4. **Install CLIP**

```Bash
cd ../
git clone https://github.com/openai/CLIP.git
python ./CLIP/setup.py install

```

## Usage



### Train

```Bash
cd RALIP/codes
bash scripts/train.sh ./cfg/bird.yml
```

### Test

```Bash
bash scripts/test.sh ./cfg/bird.yml
```

### Resume

**You can change ***state_epoch*** and the corresponding ***weight*** to continue training at breakpoints**

### TensorBoard

**The results are stored in TensorBoard files under ./logs**

```Bash
tensorboard --logdir your_path --port 8166
```

## Sampling

**The** *sample.ipynb* **can be used to sample**

## Result

### Visualization

![](./result.png)

### Experiments

Compare RATLIP and state-of-the-art models on FID values (the smaller, the better).
| Model   | CUB   | CelebA-tiny |
| :-------: | :-----: | :-----------: |
|[AttnGAN ](https://arxiv.org/abs/1711.10485)| 23.98 | 125.98      |
|[LAFITE ](https://arxiv.org/abs/2111.13792)| 14.58 | -           |
|[DF-GAN ](https://github.com/tobran/DF-GAN)| 14.81 | 137.60      |
|[GALIP ](https://github.com/tobran/GALIP)| 10.00 | 94.45       |
| **Ours**  | **13.28** | **81.48**      |

Compare RATLIP and state-of-the-art models on CLIP score values (the bigger, the better).

| Model   | CUB   | CelebA-tiny | Oxford |
| :-------: | :-----: | :----------: | :------: |
|[AttnGAN ](https://arxiv.org/abs/1711.10485)| -     | 21.15       | -      |
|[LAFITE ](https://arxiv.org/abs/2111.13792)| 31.25 | -           | -      |
|[DF-GAN ](https://github.com/tobran/DF-GAN)|29.20 | 26.67       | 24.41  |
|[GALIP ](https://github.com/tobran/GALIP)| 31.60 | 31.77       | 27.95  |
| **Ours**  | **32.03** | **31.94**    | **28.91**  |

## Citation

```Bash
@misc{lin2024ratlip,
      title={RATLIP: Generative Adversarial CLIP Text-to-Image Synthesis Based on Recurrent Affine Transformations}, 
      author={Chengde Lin and Xijun Lu and Guangxi Chen},
      year={2024},
      eprint={2405.08114},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement

* This code is adapted from [GALIP](https://github.com/tobran/GALIP) and [RAT-GAN](https://github.com/senmaoy/RAT-GAN).
* We thank Ming Tao, Bing-Kun Bao and Senmao Ye for their elegant and efficient code base.
