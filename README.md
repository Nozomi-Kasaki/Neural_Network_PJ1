# NumPy MNIST MLP/CNN Experiments

This project implements a small neural network training framework from scratch with NumPy. It trains and compares an MLP baseline and a CNN on MNIST, including learning-rate scheduler experiments, data augmentation experiments, weight visualization, and error analysis.

The project does not use PyTorch or TensorFlow. Forward propagation, backward propagation, optimization, loss functions, convolution, and learning-rate scheduling are implemented manually.

## Requirements

Install the dependencies with:

```powershell
pip install -r requirements.txt
```

Required packages:

```text
numpy
matplotlib
tqdm
```

## Dataset

The code expects MNIST gzip files under:

```text
dataset/MNIST/
```

Required files:

```text
train-images-idx3-ubyte.gz
train-labels-idx1-ubyte.gz
t10k-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz
```

The dataset is ignored by `.gitignore` and should not be uploaded to GitHub.

## File Overview

### Core Framework

`mynn/op.py`

Defines core neural network operators:

- `Linear`: fully connected layer
- `conv2D`: 2D convolution layer implemented with `im2col + matrix multiplication`
- `ReLU`: activation layer
- `Flatten`: reshapes convolution outputs before fully connected layers
- `MultiCrossEntropyLoss`: softmax + multi-class cross entropy loss
- `softmax`: stable softmax implementation

`mynn/models.py`

Defines model structures:

- `Model_MLP`: baseline MLP model, currently `784 -> 600 -> 10`
- `Model_CNN`: CNN model, currently `Conv16 -> Conv32 -> Conv64 -> FC128 -> FC10`
- Includes model saving and loading logic through `pickle`

`mynn/optimizer.py`

Defines optimizers:

- `SGD`: stochastic gradient descent with optional weight decay
- `MomentGD`: reserved placeholder

`mynn/lr_scheduler.py`

Defines learning-rate schedulers:

- `StepLR`: decay learning rate every fixed number of iterations
- `MultiStepLR`: decay learning rate at specified milestones
- `ExponentialLR`: decay learning rate smoothly at every iteration

`mynn/runner.py`

Training and evaluation helper:

- Mini-batch training loop
- Periodic evaluation
- Best model saving
- Metric and loss history recording

`mynn/metric.py`

Defines evaluation metrics:

- `accuracy`: classification accuracy

`mynn/__init__.py`

Package entry point for importing framework modules.

### Main Scripts

`test_train.py`

Main training script for MLP or CNN.

Edit:

```python
MODEL_TYPE = 'MLP'
```

or:

```python
MODEL_TYPE = 'CNN'
```

Then run:

```powershell
python test_train.py
```

The script uses a shared training budget for fair comparison:

- `10` epochs
- `batch_size = 64`
- `StepLR(step_size=2000, gamma=0.5)`
- same train/validation split
- same loss function and optimizer type

Saved models:

```text
best_models_mlp/best_model.pickle
best_models_cnn/best_model.pickle
```

`test_model.py`

Evaluates a trained MLP or CNN on both the training split and the official test set.

Edit:

```python
MODEL_TYPE = 'MLP'
```

or:

```python
MODEL_TYPE = 'CNN'
```

Then run:

```powershell
python test_model.py
```

The script prints:

- train accuracy
- test accuracy

For CNN, evaluation is batch-safe and input images are reshaped to `[N, 1, 28, 28]`.

`weight_visualization.py`

Visualizes learned model weights.

For MLP:

- visualizes first-layer hidden-unit templates
- visualizes final linear layer weights

For CNN:

- visualizes first convolution filters
- visualizes final linear layer weights

Run:

```powershell
python weight_visualization.py
```

### Experiment Scripts

`compare_schedulers.py`

Compares three learning-rate schedulers for either MLP or CNN:

- `StepLR`
- `MultiStepLR`
- `ExponentialLR`

Edit:

```python
MODEL_TYPE = 'MLP'
```

or:

```python
MODEL_TYPE = 'CNN'
```

Then run:

```powershell
python compare_schedulers.py
```

Outputs:

- test loss curves
- test accuracy curves
- saved comparison figure under `scheduler_compare_figs/`
- saved best models under `scheduler_compare_models/`

`compare_augmentations.py`

Compares separate data augmentation strategies:

- no augmentation
- rotation
- translation
- scaling

Only training images are augmented. Test images remain unchanged.

Run:

```powershell
python compare_augmentations.py
```

Outputs are saved under:

```text
augmentation_compare_figs/
augmentation_compare_models/
```

`compare_mixed_augmentation.py`

Compares no augmentation against a mixed augmentation strategy.

Mixed augmentation uses:

- `60%` unchanged
- `15%` light rotation
- `15%` light translation
- `10%` light scaling

Run:

```powershell
python compare_mixed_augmentation.py
```

Outputs are saved under:

```text
mixed_augmentation_figs/
mixed_augmentation_models/
```

`save_cnn_val_errors.py`

Loads the best CNN model and saves validation-set misclassified samples.

Run:

```powershell
python save_cnn_val_errors.py
```

Outputs:

```text
cnn_validation_errors/errors.csv
cnn_validation_errors/images/
cnn_validation_errors/error_grid.png
```

This is useful for analyzing which samples are still hard for the CNN model.

### Other Files

`dataset_explore.ipynb`

Notebook for exploring MNIST images and labels.

`hyperparameter_search.py`

Placeholder for custom hyperparameter search.

`draw_tools/plot.py`

Utility for plotting training and evaluation curves from a runner.

`draw_tools/draw.py`

Simple drawing GUI utility.

`requirements.txt`

Python dependency list.

`.gitignore`

Ignores datasets, trained models, generated figures, temporary folders, cache files, and large experiment outputs.

## Typical Workflow

1. Put MNIST gzip files under `dataset/MNIST/`.

2. Train a model:

```powershell
python test_train.py
```

3. Evaluate the trained model:

```powershell
python test_model.py
```

4. Visualize learned weights:

```powershell
python weight_visualization.py
```

5. Run scheduler comparison:

```powershell
python compare_schedulers.py
```

6. Run mixed data augmentation comparison:

```powershell
python compare_mixed_augmentation.py
```

7. Analyze CNN validation errors:

```powershell
python save_cnn_val_errors.py
```

## Notes About GitHub Upload

Do not upload datasets, trained weights, generated figures, or temporary experiment outputs. These files are ignored by `.gitignore`.

Before committing, check:

```powershell
git status
```

Make sure paths such as `dataset/`, `best_models_cnn/`, `best_models_mlp/`, `*.pickle`, and experiment output folders are not staged.
