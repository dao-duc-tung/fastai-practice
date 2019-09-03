# Data Notes

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

## Cleaning Data

- Use FileDeleter to show the top loss images for verifying/deleting them.
- After cleaning, retrain, it's normal if the model is 0.01% better.
- It's fine, just make sure we don't have too much noise in dataset.

## How much data?

- Most of time, we need less data than we think.
- Get it more if you're not satisfied with accuracy.

## How (and why) to create a good validation set

- https://www.fast.ai/2017/11/13/validation-sets/
- [**TRICK**] Train set = Train + Valid set (in Kaggle)

## Unbalanced data

- 200 black bears, 50 teddies? -> It still works!

# Training Notes

## Problem Types

- [Image Classification](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson1-pets.ipynb)
  - Data: ImageDataBunch
  - Learner: cnn_learner
- [Multi-label Image Classification](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-planet.ipynb)
  - Data:
    - ImageFileList
    - .from_folder
    - .label_from_csv
    - .random_split_by_pct
    - .datasets
    - .transform
    - .databunch
    - .normalize
  - Learner: cnn_learner
- [Image Segmentation](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson3-camvid.ipynb)
  - Data:
    - SegmentationItemList
    - .from_folder
    - .split_by_fname_file
    - .label_from_func
    - .transform
    - .databunch
    - .normalize
  - Learner: unet_learner
- [Regression](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-head-pose.ipynb)
  - Data:
    - ImageItemList
    - .from_folder
    - .split_by_valid_func
    - .label_from_func
    - .transform
    - .databunch
    - .normalize
  - Learner: cnn_learner
- [Natural Language Processing](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-imdb.ipynb)
  - Data:
    - TextList
    - .from_folder
    - .filter_by_folder
    - .split_by_rand_pct
    - .label_for_lm
    - .databunch
  - Learner: language_model_learner (train encoder)
  - Data:
    - TextList
    - .from_folder
    - .split_by_folder
    - .label_from_folder
    - .databunch
  - Learner: text_classifier_learner (train text classifier)
- [Tabular](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson4-tabular.ipynb)
  - Data:
    - TabularList
    - .from_df
    - .split_by_idx
    - .label_from_df
    - .add_test
    - .databunch
  - Learner: tabular_learner
  - People often use logistic regression, random forests, or gradient bossting machine for tabular data.
  But using neural nets nowadays tends to be reliable and effective.
- [Collaborative Filtering](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson4-collab.ipynb)
  - Data:
    - CollabDataBunch
    - .from_df
  - Learner: collab_learner

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

## Activation function

- Is an element-wise function. It does calculation on every elements.

## Adam - Do both momentum and RMSProp

- These optimizes are called **Dynamic Learning Rate**.

![](images/47.png)

Ex: When LR is small, momentum is large.

## Affine function

- Is Matrix Multiplication (same)
- Is Linear function

## Back Propagation

- Update weights matrixes/params

```python
# update param by substracting it from
# product of its gradient and learning rate
param -= (lr * param.grad())
```

## Bias

![](images/12.png)

- In Collaborative Filtering system, maybe there are certain movies that everybody likes more.
Maybe there are some users that just tend to like movies more.
- There're also some cases that if a movie doesn't have a feature that a user likes, but the movie
is very good and others like it. Then we need some ways to say "unless if it's that movie".
- So I want to add a single number of like how popular is this movie, and add a single number of
like how much does this user like movies in general. So those are called `bias` terms.
- We don't just want to have prediction equals dot product of these two things, we want to say
it's the dot product of those two things plus a bias term for a movie plus a bias term for user ID.
- Now each movie can have an overall "this is a great movie" vs "this isn't a great movie" or every
user can have an overall "this user rates movie highly" or "this user doesn't rate movies highly".

## Cold Start Problem in collaborative filtering | Recommendation System

- Recommend movies for a new user, or recommend a new movie for users.
At this point, we don't have any data in our collaborative filtering system.
- Solution 1: Ask user about some movies if he likes it
- Solution 2: Have a second model which is NOT a collaborative filtering model
but a METADATA driven model for new users or new movies.
Ex: If you're selling products and you don't want to show them a big selection
of your products and say did you like this because you just want them to buy.
You can instead try and use a metadata based tabular model what geography did they come from,
their age and sex, you can try and make some guesses about the initial recommendations.

## Cross Entropy Loss function

```python
if y == 1: return -log(y_hat)
else: return -log(1 - y_hat)
```

- MSE is not good when we need sth where predicting the right thing correctly and confidently
should have very little loss; predicting the wrong thing confidently should have a lot of loss.
Ex: `mse(4,3)` is still small --> it should be big!

## Embedding

![](images/14.png)

- Is a matrix of weights
- In this case, we have an embedding matrix for a user and an embedding matrix for a movie.

## Encoder

- Encoder in NLP model is responsible for creating, updating hidden states (understand sentences)
- Encoder is the first half of NLP model. The second half is all about prediting the next word.

## Find Learning Rate

- Sometimes the tricks to choose LR below don't work. We should try 10x less, 10x more and see
what looks best.

## Find Learning Rate before unfreezing | freezing | frozen

![](images/n1.png)

- Find the thing with the steepest slope, somewhere around 1e-2.
- In this case, choose slice(1e-2)

![](images/n3.png)

- LR = (Be4 it shoots up)/10 = 1e-2/2

## Find Learning Rate after unfreezing | unfreeze

![](images/13.png)

- Fine the strongest downward
- Min LR = on the top of that slope, shift a bit to the right
- Max LR = (LR at frozen)/10 or /5
- In this case, choose slice(3e-5, 3e-4)

![](images/n2.png)
- In this case:
- Min LR = (Be4 it shoots up = 1e-4)/10 = 1e-5
- Max LR = (LR at frozon)/5

## Latent factor | Latent feature

- In collaborative filtering user&movie system, we multiply 2 matrixes.
- If there's a number in embedding of user is high, and the corresponding number of it in embedding
of movie is also high, then the result of their product is high too. It means the movie has a
feature that that user likes.
- We don't decide the rows mean anything. But the only way that this gradient descent could come up
with a good answer is if it figures out what the aspects of movie taste are and the corresponding
features of movie are.
- The underlying features are called latent features. They are hidden things. Once we train this
neural net, they suddenly appear.

## Logistic Regression Model

- Is a neural net with NO HIDDEN LAYERS, it's a one layer neural net NO NONLINEARITIES.

## Loss function

- The thing that's telling you how far away or how close you are to the correct answer.
- For classification problems, we use cross entropy loss, known as negative log likelihood loss.
This penalizes incorrect confident predictions, and correct unconfident predictions.

## Matrix Factorization

- Is a class of collaborative filtering algorithms used in recommendation system.
- Works by decomposing the user-item interaction matrix into the product of
2 lower dimension matrixes.
- Number of factors is the width of the embedding matrix.
- Why number of factors = 40? Try few things and see what looks best.

## Mean Squared Error - Loss function

```python
def mse(y_hat, y): return ((y_hat-y)**2).mean()
```

## Mini-batches

- The only difference between stochastic gradient descent and gradient descent is something called mini-batches.
- We calculated the value of the loss on the whole dataset on every iteration.
But if your dataset is 1.5 million images in ImageNet, that's going to be really slow.
- What we do is we grab 64 images or so at a time at random, and we calculate the loss on those 64 images, and we update our weights.
- Mini-batches: A random bunch of points that you use to update your weights

## Momentum - Keep track of EWMA of step

- Previously, we have: <img src="https://latex.codecogs.com/gif.latex?w_{t}=w_{t-1}-lr\times&space;\frac{dL}{dw_{t-1}}" title="w_{t}=w_{t-1}-lr\times \frac{dL}{dw_{t-1}}" />
- Now, we have: <img src="https://latex.codecogs.com/gif.latex?w_{t}=w_{t-1}-lr\times&space;S_{t}" title="w_{t}=w_{t-1}-lr\times S_{t}" />
  - With <img src="https://latex.codecogs.com/gif.latex?S_{t}&space;=&space;\alpha\times&space;\frac{dL}{d_{w_{t-1}}}&space;+&space;(1-\alpha)\times&space;S_{t-1}" title="S_{t} = \alpha\times \frac{dL}{d_{w_{t-1}}} + (1-\alpha)\times S_{t-1}" />
- <img src="https://latex.codecogs.com/gif.latex?S_{t}" title="S_{t}" /> is called Exponentially
weighted moving average, because <img src="https://latex.codecogs.com/gif.latex?(1-\alpha)" title="(1-\alpha)" />
are going to multiply with <img src="https://latex.codecogs.com/gif.latex?S_{t-2}" title="S_{t-2}" />,
<img src="https://latex.codecogs.com/gif.latex?S_{t-3}" title="S_{t-3}" /> ...
- <img src="https://latex.codecogs.com/gif.latex?(1-\alpha)" title="(1-\alpha)" /> is momentum part,
often has value of 0.9 --> <img src="https://latex.codecogs.com/gif.latex?\alpha=0.1" title="\alpha=0.1" />

## Neural Network | Architecture

![](images/19.png)

- Yellow: `Parameters`: Number that are stored to make a calculation. Ex: Numbers inside matrixes
- Purple & Blue: `Activations`: Result of a calculation, numbers that are calculated.
Ex: result of a matrix multiply or an activation function (ReLU's output)
- Red Arrows: `Layers`: Things that do a calculation.
There're 2 types of layers (except input/output layers):
  - Yellow: Layers contain params
  - Blue: Layers contain activations
- `ReLU`: an activation function.

## Parameters | Coefficients | Weights

- Numbers that you are updating.

## plot_losses in fastai | plot losses

![](images/50.png)

- When we plot losses in fastai, it doesn't look like above. It looks like below

![](images/48.png)

- Because fastai calculates the EWMA of the losses --> easier to read chart.

## Production

- Let CPU predict in production.
- CPU can do lots of things at the same time, but not GPU.
- Use PythonAnyWhere, Zeit, Render.com for free hosting

## RMSProp - Keep track of EWMA of the gradient squared

- We have: <img src="https://latex.codecogs.com/gif.latex?w_{t}=w_{t-1}-&space;lr\times&space;\frac{dL}{d_{w_{t-1}}}\times&space;\frac{1}{\sqrt{R_{t-1}}}" title="w_{t}=w_{t-1}- lr\times \frac{dL}{d_{w_{t-1}}}\times \frac{1}{\sqrt{R_{t-1}}}" />
  - With: <img src="https://latex.codecogs.com/gif.latex?R_{t-1}=\bigg(\frac{dL}{d_{w_{t}}}\bigg)^2\times&space;\alpha+&space;(1-\alpha)\times&space;R_{t-2}" title="R_{t-1}=\bigg(\frac{dL}{d_{w_{t}}}\bigg)^2\times \alpha+ (1-\alpha)\times R_{t-2}" />

## Softmax - Activation function

- <img src="https://latex.codecogs.com/gif.latex?\sigma(z)_{i}=\frac{e^{z_{i}}}{\sum_{j=1}^{K}e^{z_{j}}}&space;\text{&space;for&space;}i=1...K\text{&space;and&space;}z=(z_{1}...z_{K})\in\mathbb{R}^K" title="\sigma(z)_{i}=\frac{e^{z_{i}}}{\sum_{j=1}^{K}e^{z_{j}}} \text{ for }i=1...K\text{ and }z=(z_{1}...z_{K})\in\mathbb{R}^K" />
- When we do single-label multi-class classification, we want `softmax` as our activation function
and cross-entropy as loss function, because these things go together in such friendly ways.
`Pytorch` will do this for us.
- The `nn.CrossEntropyLoss` is not really just cross-entropy loss, it's actually `softmax` then
cross-entropy loss.

## Transfer Learning in FastAI

![](images/2.png)

- What happened to the last layer?
  - In ResNet34, last layer has 1000 cols (1000 classes in ImageNet).
  fastai replaces it with 2 new weight matrixes with a ReLU in between
  - The first matrix has some default sizes. The second one has 'data.c' cols
- Freezing layers
  - When calling fit 1st time, it means don't back propagate the gradients back to those other
  layers except the 2 new ones.
- Unfreezing and Using Discriminative Learning Rates
  - After a while, we say "this model is looking pretty good. We should train the rest of the
  network now" --> We unfreeze.
  - But we know that: "The new layers need more training, and the ones at the start don't need
  much training" --> We split out model into a few sections and give them different LRs.
  --> This process is called Using Discriminative Learning Rates.
  - In `fit`/`fit_one_cycle` function params, we put the LR as a number/slice(number)/ or
  slice(number, number)
    - Number: every layer get the same LR --> not using discriminative LRs
    - slice(Number): the final layers get Number, the others get Number/3
    - slice(Num1, Num2): the 1st group of layers get Num1, second get (Num1+Num2)/2, third get Num2.
    By default, there're 3 groups: 3rd one is the new layers, 1st and 2nd ones are splitted half
    of the rest.

## Underfitting | Overfitting

![](images/40.png)

## Universal Approximation Theorem

- If we have enough weight matrixes, it can solve any arbitrarily complex mathematical function
to any arbitrarily high level of accuracy (assuming we can train params in terms of time and
data availability...)

## Weight Decay - Penalize the complexity of model

- Previously, we update weights by using <img src="https://latex.codecogs.com/gif.latex?w_{t}=w_{t-1}-lr\times&space;\frac{dL}{dw_{t-1}}" title="w_{t}=w_{t-1}-lr\times \frac{dL}{dw_{t-1}}" />
  - With <img src="https://latex.codecogs.com/gif.latex?L(x,w)=mse(m(x,w),y)&plus;wd\cdot&space;\sum&space;w^{2}" title="L(x,w)=mse(m(x,w),y)+wd\cdot \sum w^{2}" />
  - And `m` is our model
- Then, we have: <img src="https://latex.codecogs.com/gif.latex?\frac{dL}{dw}&space;=&space;\frac{d}{dw}wd\cdot&space;w^{2}&space;=&space;2wd\cdot&space;w" title="\frac{dL}{dw} = \frac{d}{dw}wd\cdot w^{2} = 2wd\cdot w" />
  - Remove Sigma sum symbol is okay.
- All weight decay does is it subtracts some constant times the weights every time we do a batch.
We can replace `wd` with `2wd` without loss of generality.
  - When it's (<img src="https://latex.codecogs.com/gif.latex?wd\cdot&space;w^{2}" title="wd\cdot w^{2}" />)
  where we add the square to Loss function --> it's called **L2 Regularization**.
  - When it's (<img src="https://latex.codecogs.com/gif.latex?wd\cdot&space;w" title="wd\cdot w" />)
  where we subtract <img src="https://latex.codecogs.com/gif.latex?wd" title="wd" /> times weights
  from the gradients --> it's called **weight decay**.
- By using WD, we can make giant NN and still avoid overfitting, or really small datasets with
moderately large sized models and avoid overfitting.
- Sometimes, you might still find you don't have enough data in cases where you're not overfitting
by adding lots of weight decay and it's just not training very well.

## Y Range in collaborative filtering

- Use sigmoid to make result in (0, 1)
- User y_range=[0, 5.5] because there's movie with 5 stars, but sigmoid never get to 5.

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
