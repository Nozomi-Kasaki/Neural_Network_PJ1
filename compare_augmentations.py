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
MODEL_TYPE = 'MLP'   # choose from {'MLP', 'CNN'}
SEED = 309
VALID_SIZE = 10000
IDX_PATH = r'.\idx.pickle'

COMMON_BATCH_SIZE = 64
COMMON_NUM_EPOCHS = 10
COMMON_LOG_ITERS = 100
COMMON_EVAL_ITERS = 200
COMMON_STEP_SIZE = 2000
COMMON_GAMMA = 0.5
EVAL_BATCH_SIZE = 256

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

AUGMENTATIONS = ['none', 'rotation', 'translation', 'scaling']
ROTATION_DEGREES = 10.0
TRANSLATION_PIXELS = 2.0
SCALE_RANGE = (0.9, 1.1)

FIG_SAVE_DIR = r'.\augmentation_compare_figs'
MODEL_SAVE_DIR = r'.\augmentation_compare_models'
SHOW_PLOT = True

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'
test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

AUGMENTATION_STYLES = {
    'none': {'color': '#4C78A8', 'linestyle': '-', 'linewidth': 1.2},
    'rotation': {'color': '#F58518', 'linestyle': '--', 'linewidth': 1.2},
    'translation': {'color': '#54A24B', 'linestyle': '-.', 'linewidth': 1.2},
    'scaling': {'color': '#B279A2', 'linestyle': ':', 'linewidth': 1.4},
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

    rng = np.random.RandomState(SEED)
    idx = rng.permutation(np.arange(num_samples))
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
        test_inputs = test_imgs.reshape(-1, 1, rows, cols)
    else:
        test_inputs = test_imgs

    num_classes = int(train_labs.max()) + 1
    flat_dim = rows * cols
    return (train_imgs, train_labs), (test_inputs, test_labs), num_classes, flat_dim, rows, cols


def sample_bilinear(images, x, y):
    batch_size, height, width = images.shape

    x0 = np.floor(x).astype(np.int64)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int64)
    y1 = y0 + 1

    x0_clip = np.clip(x0, 0, width - 1)
    x1_clip = np.clip(x1, 0, width - 1)
    y0_clip = np.clip(y0, 0, height - 1)
    y1_clip = np.clip(y1, 0, height - 1)
    batch_idx = np.arange(batch_size)[:, None, None]

    top_left = images[batch_idx, y0_clip, x0_clip]
    top_right = images[batch_idx, y0_clip, x1_clip]
    bottom_left = images[batch_idx, y1_clip, x0_clip]
    bottom_right = images[batch_idx, y1_clip, x1_clip]

    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    valid_a = (x0 >= 0) & (x0 < width) & (y0 >= 0) & (y0 < height)
    valid_b = (x1 >= 0) & (x1 < width) & (y0 >= 0) & (y0 < height)
    valid_c = (x0 >= 0) & (x0 < width) & (y1 >= 0) & (y1 < height)
    valid_d = (x1 >= 0) & (x1 < width) & (y1 >= 0) & (y1 < height)

    sampled = (
        wa * top_left * valid_a +
        wb * top_right * valid_b +
        wc * bottom_left * valid_c +
        wd * bottom_right * valid_d
    )
    return sampled.astype(np.float32)


def transform_images(images, augmentation, rng):
    if augmentation == 'none':
        return images

    batch_size, height, width = images.shape
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing='ij'
    )
    xx = xx[None, :, :]
    yy = yy[None, :, :]
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    x_centered = xx - center_x
    y_centered = yy - center_y

    if augmentation == 'rotation':
        angles = rng.uniform(-ROTATION_DEGREES, ROTATION_DEGREES, size=batch_size)
        angles = np.deg2rad(angles).astype(np.float32)[:, None, None]
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        x_src = cos_a * x_centered + sin_a * y_centered + center_x
        y_src = -sin_a * x_centered + cos_a * y_centered + center_y
    elif augmentation == 'translation':
        dx = rng.uniform(-TRANSLATION_PIXELS, TRANSLATION_PIXELS, size=batch_size).astype(np.float32)[:, None, None]
        dy = rng.uniform(-TRANSLATION_PIXELS, TRANSLATION_PIXELS, size=batch_size).astype(np.float32)[:, None, None]
        x_src = xx - dx
        y_src = yy - dy
    elif augmentation == 'scaling':
        scale = rng.uniform(SCALE_RANGE[0], SCALE_RANGE[1], size=batch_size).astype(np.float32)[:, None, None]
        x_src = x_centered / scale + center_x
        y_src = y_centered / scale + center_y
    else:
        raise ValueError(f'Unsupported augmentation: {augmentation}')

    return sample_bilinear(images, x_src, y_src)


def prepare_train_batch(batch_flat, model_type, rows, cols, augmentation, rng):
    batch_images = batch_flat.reshape(-1, rows, cols)
    batch_images = transform_images(batch_images, augmentation, rng)
    if model_type == 'CNN':
        return batch_images.reshape(-1, 1, rows, cols)
    return batch_images.reshape(batch_images.shape[0], rows * cols)


def evaluate(model, loss_fn, test_set):
    test_inputs, test_labels = test_set
    total_loss = 0.0
    total_correct = 0
    total_samples = test_labels.shape[0]

    for start in range(0, total_samples, EVAL_BATCH_SIZE):
        end = start + EVAL_BATCH_SIZE
        batch_x = test_inputs[start:end]
        batch_y = test_labels[start:end]
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        preds = np.argmax(logits, axis=-1)

        total_loss += float(loss) * batch_y.shape[0]
        total_correct += np.sum(preds == batch_y)

    return total_correct / total_samples, total_loss / total_samples


def train_once(model_type, augmentation, train_set, test_set, num_classes, flat_dim, rows, cols):
    np.random.seed(SEED)
    model = build_model(model_type, num_classes, flat_dim)
    optimizer = nn.optimizer.SGD(init_lr=get_init_lr(model_type), model=model)
    scheduler = nn.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=COMMON_STEP_SIZE,
        gamma=COMMON_GAMMA
    )
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=num_classes)
    aug_rng = np.random.RandomState(SEED + AUGMENTATIONS.index(augmentation) + 1)

    train_imgs, train_labs = train_set
    num_batches = int(np.ceil(train_imgs.shape[0] / COMMON_BATCH_SIZE))
    global_iteration = 0
    best_score = 0.0
    test_loss_curve = []
    test_score_curve = []

    save_dir = os.path.join(MODEL_SAVE_DIR, model_type.lower(), augmentation)
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(COMMON_NUM_EPOCHS):
        shuffle_rng = np.random.RandomState(SEED + epoch)
        idx = shuffle_rng.permutation(train_imgs.shape[0])
        shuffled_imgs = train_imgs[idx]
        shuffled_labs = train_labs[idx]

        for iteration in range(num_batches):
            start = iteration * COMMON_BATCH_SIZE
            end = start + COMMON_BATCH_SIZE
            batch_flat = shuffled_imgs[start:end]
            batch_y = shuffled_labs[start:end]

            batch_x = prepare_train_batch(batch_flat, model_type, rows, cols, augmentation, aug_rng)
            logits = model(batch_x)
            train_loss = loss_fn(logits, batch_y)
            train_score = nn.metric.accuracy(logits, batch_y)

            loss_fn.backward()
            optimizer.step()
            scheduler.step()

            should_eval = (
                global_iteration % COMMON_EVAL_ITERS == 0 or
                iteration == num_batches - 1
            )
            if should_eval:
                test_score, test_loss = evaluate(model, loss_fn, test_set)
                test_score_curve.append((global_iteration, test_score))
                test_loss_curve.append((global_iteration, test_loss))
                if test_score > best_score:
                    model.save_model(os.path.join(save_dir, 'best_model.pickle'))
                    best_score = test_score

            if iteration % COMMON_LOG_ITERS == 0:
                print(f"epoch: {epoch}, iteration: {iteration}, augmentation: {augmentation}")
                print(f"[Train] loss: {train_loss}, score: {train_score}")
                if len(test_score_curve) > 0:
                    print(f"[Test] loss: {test_loss_curve[-1][1]}, score: {test_score_curve[-1][1]}")

            global_iteration += 1

    return {
        'test_loss': test_loss_curve,
        'test_score': test_score_curve,
        'best_score': best_score,
    }


def plot_augmentation_curves(model_type, histories):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for augmentation, history in histories.items():
        loss_steps = [item[0] for item in history['test_loss']]
        loss_values = [item[1] for item in history['test_loss']]
        score_steps = [item[0] for item in history['test_score']]
        score_values = [item[1] for item in history['test_score']]
        style = AUGMENTATION_STYLES[augmentation]

        axes[0].plot(loss_steps, loss_values, label=augmentation, **style)
        axes[1].plot(score_steps, score_values, label=augmentation, **style)

    axes[0].set_title(f'{model_type} Test Loss')
    axes[0].set_xlabel('iteration')
    axes[0].set_ylabel('loss')
    axes[0].legend()

    axes[1].set_title(f'{model_type} Test Accuracy')
    axes[1].set_xlabel('iteration')
    axes[1].set_ylabel('accuracy')
    axes[1].legend()

    fig.suptitle(f'{model_type} Data Augmentation Comparison ({COMMON_NUM_EPOCHS} epochs)')
    fig.tight_layout()
    return fig


def main():
    train_set, test_set, num_classes, flat_dim, rows, cols = prepare_datasets(MODEL_TYPE)
    histories = {}

    for augmentation in AUGMENTATIONS:
        print(f'\n===== {MODEL_TYPE} with {augmentation} augmentation =====')
        histories[augmentation] = train_once(
            MODEL_TYPE,
            augmentation,
            train_set,
            test_set,
            num_classes,
            flat_dim,
            rows,
            cols
        )

    fig = plot_augmentation_curves(MODEL_TYPE, histories)
    os.makedirs(FIG_SAVE_DIR, exist_ok=True)
    fig_path = os.path.join(
        FIG_SAVE_DIR,
        f'{MODEL_TYPE.lower()}_augmentation_comparison_{COMMON_NUM_EPOCHS}epochs.png'
    )
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f'\nFigure saved to: {fig_path}')

    for augmentation, history in histories.items():
        print(f"{augmentation} best test accuracy: {history['best_score']:.6f}")

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()
