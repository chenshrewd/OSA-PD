# OSA-PD

### Train
```bash
python main.py --query-strategy OSA-PRD --init-percent 8 --known-class 20 --dataset cifar100 --model resnet_cifar

python main.py --query-strategy OSA-PRD --init-percent 1 --known-class 2 --dataset cifar10 --model resnet_cifar

python main.py --query-strategy OSA-PRD --init-percent 8 --known-class 40 --dataset tinyimagenet --model resnet18
```