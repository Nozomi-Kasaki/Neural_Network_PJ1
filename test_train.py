# An example of reading in the data and training the model.
# Switch MODEL_TYPE to compare MLP and CNN under the same training pipeline.
import gzip
import pickle
from struct import unpack

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn
from draw_tools.plot import plot


# ------------------------------
# Experiment config
# ------------------------------
MODEL_TYPE = 'CNN'   # choose from {'MLP', 'CNN'}
SEED = 309
VALID_SIZE = 10000
SAVE_DIR = f'./best_models_{MODEL_TYPE.lower()}'

# Shared training budget for a fair comparison.
COMMON_BATCH_SIZE = 64
COMMON_NUM_EPOCHS = 10
COMMON_LOG_ITERS = 100
COMMON_EVAL_ITERS = 200
COMMON_STEP_SIZE = 2000
COMMON_GAMMA = 0.5

# Model-specific config
MLP_HIDDEN_DIM = 600
MLP_WEIGHT_DECAY = [1e-4, 1e-4]
MLP_INIT_LR = 0.03

CNN_CHANNELS = (16, 32, 64)
CNN_KERNEL_SIZE = 3
CNN_STRIDES = (1, 2, 2)
CNN_PADDING = 1
CNN_FC_HIDDEN_DIM = 128
CNN_WEIGHT_DECAY = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4]
CNN_INIT_LR = 0.01

# Dataset path
train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'


def build_model(model_type, num_classes, flat_dim):
    if model_type == 'MLP':
        return nn.models.Model_MLP(
            [flat_dim, MLP_HIDDEN_DIM, num_classes],
            'ReLU',
            MLP_WEIGHT_DECAY
        )
    if model_type == 'CNN':
        return nn.models.Model_CNN(
            input_shape=(1, 28, 28),
            num_classes=num_classes,
            conv_channels=CNN_CHANNELS,
            kernel_size=CNN_KERNEL_SIZE,
            conv_strides=CNN_STRIDES,
            padding=CNN_PADDING,
            fc_hidden_dim=CNN_FC_HIDDEN_DIM,
            lambda_list=CNN_WEIGHT_DECAY
        )
    raise ValueError(f'Unsupported MODEL_TYPE: {model_type}')


if MODEL_TYPE == 'MLP':
    INIT_LR = MLP_INIT_LR
elif MODEL_TYPE == 'CNN':
    INIT_LR = CNN_INIT_LR
else:
    raise ValueError(f'Unsupported MODEL_TYPE: {MODEL_TYPE}')

BATCH_SIZE = COMMON_BATCH_SIZE
NUM_EPOCHS = COMMON_NUM_EPOCHS
LOG_ITERS = COMMON_LOG_ITERS
EVAL_ITERS = COMMON_EVAL_ITERS
STEP_SIZE = COMMON_STEP_SIZE
GAMMA = COMMON_GAMMA


np.random.seed(SEED)

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

# Choose samples from the training set as validation set.
idx = np.random.permutation(np.arange(num))
with open('idx.pickle', 'wb') as f:
    pickle.dump(idx, f)

train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:VALID_SIZE]
valid_labs = train_labs[:VALID_SIZE]
train_imgs = train_imgs[VALID_SIZE:]
train_labs = train_labs[VALID_SIZE:]

# Normalize from [0, 255] to [0, 1].
train_imgs = train_imgs.astype(np.float32) / 255.0
valid_imgs = valid_imgs.astype(np.float32) / 255.0

num_classes = int(train_labs.max()) + 1
flat_dim = train_imgs.shape[-1]

if MODEL_TYPE == 'CNN':
    train_imgs = train_imgs.reshape(-1, 1, rows, cols)
    valid_imgs = valid_imgs.reshape(-1, 1, rows, cols)

model = build_model(MODEL_TYPE, num_classes, flat_dim)
optimizer = nn.optimizer.SGD(init_lr=INIT_LR, model=model)
scheduler = nn.lr_scheduler.StepLR(optimizer=optimizer, step_size=STEP_SIZE, gamma=GAMMA)
loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=num_classes)

runner = nn.runner.RunnerM(
    model,
    optimizer,
    nn.metric.accuracy,
    loss_fn,
    batch_size=BATCH_SIZE,
    scheduler=scheduler
)

runner.train(
    [train_imgs, train_labs],
    [valid_imgs, valid_labs],
    num_epochs=NUM_EPOCHS,
    log_iters=LOG_ITERS,
    eval_iters=EVAL_ITERS,
    save_dir=SAVE_DIR
)

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.show()
