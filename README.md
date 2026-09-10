# selfsup-reverse-attack

An independent PyTorch implementation of
[Adversarial Attacks are Reversible with Natural Supervision](https://arxiv.org/abs/2103.14222)
(Mao et al., ICCV 2021). Official code: [cvlab-columbia/SelfSupDefense](https://github.com/cvlab-columbia/SelfSupDefense).

An adversarial attack fools the classifier, but it also disrupts a self supervised contrastive
loss. At test time we search for a small perturbation that restores this loss and classify the
corrected image, with no retraining and no labels.

## Setup

```bash
pip install -r requirements.txt
python tests/smoke_test.py
```

## Usage

Train the classifier and the contrastive head, then evaluate:

```bash
python train_classifier.py
python train_ssl.py --ckpt checkpoints/backbone_preactresnet18_fgsm_rs.pt
python evaluate.py --ckpt checkpoints/backbone_preactresnet18_fgsm_rs.pt --ssl-ckpt checkpoints/ssl_head.pt
python plot_results.py
```

To start from a pretrained [RobustBench](https://github.com/RobustBench/robustbench) model instead:

```bash
pip install "setuptools<82" git+https://github.com/RobustBench/robustbench.git
python prepare_robustbench.py --model Carmon2019Unlabeled
```

then use `checkpoints/backbone_robustbench_Carmon2019Unlabeled.pt` as `--ckpt`. With the paper's settings:

```bash
python train_ssl.py --ckpt checkpoints/backbone_robustbench_Carmon2019Unlabeled.pt --epochs 200 --train-per-class 5000
python evaluate.py --ckpt checkpoints/backbone_robustbench_Carmon2019Unlabeled.pt --ssl-ckpt checkpoints/ssl_head.pt \
    --attack autoattack --test-per-class 1000 --n-test 10000
```

## Citation

```bibtex
@InProceedings{Mao_2021_ICCV,
  author    = {Mao, Chengzhi and Chiquier, Mia and Wang, Hao and Yang, Junfeng and Vondrick, Carl},
  title     = {Adversarial Attacks Are Reversible With Natural Supervision},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2021},
  pages     = {661-671}
}
```
