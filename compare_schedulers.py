import gzip
import os
import pickle
from struct import unpack

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn


# ------------------------------
# Comparison config
# ------------------------------
MODEL_TYPE = 'CNN'   # choose from {'MLP', 'CNN'}
SEED = 309
VALID_SIZE = 10000
IDX_PATH = r'.\idx.pickle'
FIG_SAVE_DIR = r'.\scheduler_compare_figs'
SHOW_PLOT = True

# Compare schedulers under the same 10-epoch training budget.
COMMON_BATCH_SIZE = 64
COMMON_NUM_EPOCHS = 10
COMMON_LOG_ITERS = 100
COMMON_EVAL_ITERS = 200
COMMON_STEP_SIZE = 2000
COMMON_GAMMA = 0.5

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

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'
test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

SCHEDULER_STYLES = {
    'StepLR': {'color': '#4C78A8', 'linestyle': '-', 'linewidth': 1.2},
    'MultiStepLR': {'color': '#F58518', 'linestyle': '--', 'linewidth': 1.2},
    'ExponentialLR': {'color': '#54A24B', 'linestyle': '-.', 'linewidth': 1.2},
}


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


def get_init_lr(model_type):
    if model_type == 'MLP':
        return MLP_INIT_LR
    if model_type == 'CNN':
        return CNN_INIT_LR
    raise ValueError(f'Unsupported MODEL_TYPE: {model_type}')


def build_scheduler(name, optimizer):
    if name == 'StepLR':
        return nn.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=COMMON_STEP_SIZE,
            gamma=COMMON_GAMMA
        )
    if name == 'MultiStepLR':
        milestones = [COMMON_STEP_SIZE, COMMON_STEP_SIZE * 2, COMMON_STEP_SIZE * 3]
        return nn.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=milestones,
            gamma=COMMON_GAMMA
        )
    if name == 'ExponentialLR':
        exp_gamma = COMMON_GAMMA ** (1.0 / COMMON_STEP_SIZE)
        return nn.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=exp_gamma
        )
    raise ValueError(f'Unsupported scheduler: {name}')


def load_images_labels(images_path, labels_path):
    with gzip.open(images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)

    with gzip.open(labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    images = images.astype(np.float32) / 255.0
    return images, labels, rows, cols


def get_split_indices(num_samples):
    if os.path.exists(IDX_PATH):
        with open(IDX_PATH, 'rb') as f:
            idx = pickle.load(f)
        if len(idx) == num_samples:
            return idx

    np.random.seed(SEED)
    idx = np.random.permutation(np.arange(num_samples))
    with open(IDX_PATH, 'wb') as f:
        pickle.dump(idx, f)
    return idx


def prepare_datasets(model_type):
    train_imgs, train_labs, rows, cols = load_images_labels(train_images_path, train_labels_path)
    test_imgs, test_labs, _, _ = load_images_labels(test_images_path, test_labels_path)

    idx = get_split_indices(train_imgs.shape[0])
    train_imgs = train_imgs[idx][VALID_SIZE:]
    train_labs = train_labs[idx][VALID_SIZE:]

    if model_type == 'CNN':
        train_imgs = train_imgs.reshape(-1, 1, rows, cols)
        test_imgs = test_imgs.reshape(-1, 1, rows, cols)

    num_classes = int(train_labs.max()) + 1
    flat_dim = rows * cols
    return (train_imgs, train_labs), (test_imgs, test_labs), num_classes, flat_dim


def train_once(model_type, scheduler_name, train_set, test_set, num_classes, flat_dim):
    np.random.seed(SEED)

    model = build_model(model_type, num_classes, flat_dim)
    optimizer = nn.optimizer.SGD(init_lr=get_init_lr(model_type), model=model)
    scheduler = build_scheduler(scheduler_name, optimizer)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=num_classes)

    runner = nn.runner.RunnerM(
        model,
        optimizer,
        nn.metric.accuracy,
        loss_fn,
        batch_size=COMMON_BATCH_SIZE,
        scheduler=scheduler
    )

    save_dir = os.path.join('scheduler_compare_models', model_type.lower(), scheduler_name.lower())
    runner.train(
        train_set,
        test_set,
        num_epochs=COMMON_NUM_EPOCHS,
        log_iters=COMMON_LOG_ITERS,
        eval_iters=COMMON_EVAL_ITERS,
        save_dir=save_dir
    )
    return runner


def plot_scheduler_curves(model_type, runners):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for scheduler_name, runner in runners.items():
        loss_steps = [item[0] for item in runner.dev_loss]
        loss_values = [item[1] for item in runner.dev_loss]
        score_steps = [item[0] for item in runner.dev_scores]
        score_values = [item[1] for item in runner.dev_scores]
        style = SCHEDULER_STYLES[scheduler_name]

        axes[0].plot(loss_steps, loss_values, label=scheduler_name, **style)
        axes[1].plot(score_steps, score_values, label=scheduler_name, **style)

    axes[0].set_title(f'{model_type} Test Loss')
    axes[0].set_xlabel('iteration')
    axes[0].set_ylabel('loss')
    axes[0].legend()

    axes[1].set_title(f'{model_type} Test Accuracy')
    axes[1].set_xlabel('iteration')
    axes[1].set_ylabel('accuracy')
    axes[1].legend()

    fig.suptitle(f'{model_type} Scheduler Comparison ({COMMON_NUM_EPOCHS} epochs)')
    fig.tight_layout()
    return fig


def main():
    scheduler_names = ['StepLR', 'MultiStepLR', 'ExponentialLR']
    train_set, test_set, num_classes, flat_dim = prepare_datasets(MODEL_TYPE)
    runners = {}

    for scheduler_name in scheduler_names:
        print(f'\n===== {MODEL_TYPE} with {scheduler_name} =====')
        runner = train_once(
            MODEL_TYPE,
            scheduler_name,
            train_set,
            test_set,
            num_classes,
            flat_dim
        )
        runners[scheduler_name] = runner

    fig = plot_scheduler_curves(MODEL_TYPE, runners)
    os.makedirs(FIG_SAVE_DIR, exist_ok=True)
    fig_path = os.path.join(
        FIG_SAVE_DIR,
        f'{MODEL_TYPE.lower()}_scheduler_comparison_{COMMON_NUM_EPOCHS}epochs.png'
    )
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f'\nFigure saved to: {fig_path}')

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()
