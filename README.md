# Torch Hawk
A benchmark on popular deep learning tricks on computer vision tasks with famous datasets. 



## Installation

This project was developed in Python 3.7 with PyTorch 1.1. If you have a previous version of PyTorch, please consider switching `torch.utils.tensorboard` dependency to [tensorboardX](https://github.com/lanpa/tensorboardX).

```bash
git clone https://github.com/frgfm/TorchHawk.git
cd TorchHawk
pip install -r requirements.txt
mkdir img_classification/logs
```



## Usage

How to train your model



### Start your training



```bash
python img_classification/main.py test --dataset mnist -n 10 --lr 5e-5 --batch_size 8 --momentum 0.9 --weight_decay 5e-4 --nesterov
```



Depending on your seed, you should get similar results to:

```bash
Epoch 1/10 - Validation loss: 0.637735663574934 - Accuracy: 0.8272                                                                                              
Epoch 2/10 - Validation loss: 0.2974701254960149 - Accuracy: 0.9073                                                                                             
Epoch 3/10 - Validation loss: 0.20943492125663907 - Accuracy: 0.9384                                                                                            
Epoch 4/10 - Validation loss: 0.16496901647993364 - Accuracy: 0.9528                                                                                            
Epoch 5/10 - Validation loss: 0.12413150504280347 - Accuracy: 0.963                                                                                             
Epoch 6/10 - Validation loss: 0.11015441644398961 - Accuracy: 0.9674                                                                                            
Epoch 7/10 - Validation loss: 0.09114648424847983 - Accuracy: 0.9742                                                                                            
Epoch 8/10 - Validation loss: 0.08870566121073789 - Accuracy: 0.9721                                                                                            
Epoch 9/10 - Validation loss: 0.07673511246949201 - Accuracy: 0.9768                                                                                            
Epoch 10/10 - Validation loss: 0.07370509473233833 - Accuracy: 0.9786 
```



### Running the tensorboard interface

Start the tensorboard server locally to visualize your training losses:

```bash
tensorboard --logdir=logs
```

Then open a new tab in your browser and navigate to `<YOUR_COMPUTER_NAME>:6006`  to monitor your training.

![tb_loss](static/images/tb_loss.png)





## Submitting a request / Reporting an issue

Regarding issues, use the following format for the title:

> [Topic] Your Issue name

Example:

> [State saving] Add a feature to automatically save and load model states