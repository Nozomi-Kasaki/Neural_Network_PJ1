# plot the score and loss
import matplotlib.pyplot as plt

colors_set = {'Kraftime': ('#E3E37D', '#968A62')}


def plot(runner, axes, set=colors_set['Kraftime']):
    train_color = set[0]
    dev_color = set[1]

    train_steps = list(range(len(runner.train_scores)))
    dev_loss_steps = [item[0] for item in runner.dev_loss]
    dev_loss_values = [item[1] for item in runner.dev_loss]
    dev_score_steps = [item[0] for item in runner.dev_scores]
    dev_score_values = [item[1] for item in runner.dev_scores]

    axes[0].plot(train_steps, runner.train_loss, color=train_color, label="Train loss")
    axes[0].plot(dev_loss_steps, dev_loss_values, color=dev_color, linestyle="--", label="Dev loss")
    axes[0].set_ylabel("loss")
    axes[0].set_xlabel("iteration")
    axes[0].legend(loc='upper right')

    axes[1].plot(train_steps, runner.train_scores, color=train_color, label="Train accuracy")
    axes[1].plot(dev_score_steps, dev_score_values, color=dev_color, linestyle="--", label="Dev accuracy")
    axes[1].set_ylabel("score")
    axes[1].set_xlabel("iteration")
    axes[1].legend(loc='lower right')
