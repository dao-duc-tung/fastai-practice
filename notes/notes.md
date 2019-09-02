# Data Notes

## Problem Types

- [Image Classification](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson1-pets.ipynb)
  - cnn_learner
- [Multi-label Image Classification](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-planet.ipynb)
  - cnn_learner
- [Image Segmentation](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson3-camvid.ipynb)
  - unet_learner
- [Regression](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-head-pose.ipynb)
  - cnn_learner
- [Natural Language Processing](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-imdb.ipynb)
  - language_model_learner (train encoder)
  - text_classifier_learner (train text classifier)

## Cleaning Data

- Use FileDeleter to show the top loss images for verifying/deleting them.
- After cleaning, retrain, it's normal if the model is 0.01% better.
- It's fine, just make sure we don't have too much noise in dataset.

## How much data?

- Most of time, we need less data than we think.
- Get it more if you're not satisfied with accuracy.

## Unbalanced data

- 200 black bears, 50 teddies? -> It still works!

## How (and why) to create a good validation set

https://www.fast.ai/2017/11/13/validation-sets/

## Batch size for Image Segmentation problem | Free memory

```python
size = src_size//2
free = gpu_mem_get_free_no_cache()
if free > 8200: bs=8
else:           bs=4

# after that, if we want to train more with larger size, then do
learn.destroy()
size = src_size
free = gpu_mem_get_free_no_cache()
if free > 8200: bs=3
else:           bs=1
```

# Training Notes

## Statements
- Training Loss should be ALWAYS < Validation Loss
- Training/Valid Loss goes up a bit be4 goes down.
Because LR starts low, goes up, and then goes down.
- Error Rate tells you if the model is overfitting
- If you're still under fitting, then you have to decrease regularization.
Ex: weight decay, dropout, data augmentation, ...
- [**TRICK**] Make a lot of smaller datasets to step up from in tuning.
Ex: 64x64, 128x128, 256x256, ...
- [**TRICK**] U-Net is suitable for segmentation problem,
especially for Biomedical image segmentation.
- [**TRICK**] Create learner using `.to_fp16()` method for computing
in 16-bit floating point instead of 32-bit. Less precise but better result and faster training.

## Production

- Let CPU predict in production.
- CPU can do lots of things at the same time, but not GPU.
- Use PythonAnyWhere, Zeit, Render.com for free hosting

## Encoder

- Encoder in NLP model is responsible for creating, updating hidden states (understand sentences)

## Mini-batches

- The only difference between stochastic gradient descent and gradient descent is something called mini-batches.
- We calculated the value of the loss on the whole dataset on every iteration.
But if your dataset is 1.5 million images in ImageNet, that's going to be really slow.
- What we do is we grab 64 images or so at a time at random, and we calculate the loss on those 64 images, and we update our weights.
- Mini-batches: A random bunch of points that you use to update your weights

## Parameters | Coefficients | Weights

- Numbers that you are updating.

## Loss function

- The thing that's telling you how far away or how close you are to the correct answer.
- For classification problems, we use cross entropy loss, known as negative log likelihood loss.
This penalizes incorrect confident predictions, and correct unconfident predictions.

## Underfitting | Overfitting

![](images/40.png)

## Find LR before unfreezing | freezing | frozen

![](images/n1.png)

- Find the thing with the steepest slope, somewhere around 1e-2.
- In this case, choose slice(1e-2)

![](images/n3.png)

- LR = (Be4 it shoots up)/10 = 1e-2/2

## Find LR after unfreezing

![](images/13.png)

- Fine the strongest downward
- Min LR = on the top of that slope, shift a bit to the right
- Max LR = (LR at frozen)/10 or /5
- In this case, choose slice(3e-5, 3e-4)

![](images/n2.png)
- In this case:
- Min LR = (Be4 it shoots up = 1e-4)/10 = 1e-5
- Max LR = (LR at frozon)/5

## Loss function: Mean Squared Error

```python
def mse(y_hat, y): return ((y_hat-y)**2).mean()
```

## Loss function: Root Mean Squared Error

# Code Notes

## [SGD](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson2-sgd.ipynb) | Gradient

```python
def mse(y_hat, y): return ((y_hat-y)**2).mean()

a = tensor(-1.,1)
a = nn.Parameter(a); a

def update():
  y_hat = x@a
  loss = mse(y, y_hat)
  if t % 10 == 0: print(loss)
  loss.backward()
  with torch.no_grad():
    a.sub_(lr * a.grad)
    a.grad.zero_()
```
