from abc import abstractmethod
import numpy as np


def get_im2col_indices(x_shape, field_height, field_width, padding=0, stride=1):
    _, channels, height, width = x_shape
    out_height = (height + 2 * padding - field_height) // stride + 1
    out_width = (width + 2 * padding - field_width) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, channels)
    i1 = stride * np.repeat(np.arange(out_height), out_width)

    j0 = np.tile(np.arange(field_width), field_height)
    j0 = np.tile(j0, channels)
    j1 = stride * np.tile(np.arange(out_width), out_height)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(channels), field_height * field_width).reshape(-1, 1)

    return k, i, j


def im2col_indices(x, field_height, field_width, padding=0, stride=1):
    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode='constant'
    )
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    cols = x_padded[:, k, i, j]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * x.shape[1], -1)
    return cols


def col2im_indices(cols, x_shape, field_height, field_width, padding=0, stride=1):
    batch_size, channels, height, width = x_shape
    padded_height = height + 2 * padding
    padded_width = width + 2 * padding
    x_padded = np.zeros((batch_size, channels, padded_height, padded_width), dtype=cols.dtype)

    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(channels * field_height * field_width, -1, batch_size)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return np.matmul(X, self.params['W']) + self.params['b']

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        assert self.input is not None
        assert grad.shape[0] == self.input.shape[0]

        self.grads['W'] = np.matmul(self.input.T, grad)
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)

        return np.matmul(grad, self.params['W'].T)
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(size=(1, out_channels, 1, 1))
        self.params = {'W' : self.W, 'b' : self.b}
        self.grads = {'W' : None, 'b' : None}

        self.input = None
        self.input_cols = None
        self.output_shape = None

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        assert X.ndim == 4
        assert X.shape[1] == self.in_channels

        self.input = X
        batch_size, _, height, width = X.shape
        out_h = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        self.input_cols = im2col_indices(
            X,
            self.kernel_size,
            self.kernel_size,
            padding=self.padding,
            stride=self.stride
        )

        weight_cols = self.params['W'].reshape(self.out_channels, -1)
        outputs = weight_cols @ self.input_cols
        outputs += self.params['b'].reshape(self.out_channels, 1)
        outputs = outputs.reshape(self.out_channels, out_h, out_w, batch_size).transpose(3, 0, 1, 2)

        self.output_shape = (batch_size, self.out_channels, out_h, out_w)
        return outputs

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        assert self.input is not None
        grads_cols = grads.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
        weight_cols = self.params['W'].reshape(self.out_channels, -1)

        grad_W = grads_cols @ self.input_cols.T
        grad_W = grad_W.reshape(self.params['W'].shape)
        grad_b = np.sum(grads, axis=(0, 2, 3), keepdims=True)

        grad_input_cols = weight_cols.T @ grads_cols
        grad_input = col2im_indices(
            grad_input_cols,
            self.input.shape,
            self.kernel_size,
            self.kernel_size,
            padding=self.padding,
            stride=self.stride
        )

        self.grads['W'] = grad_W
        self.grads['b'] = grad_b
        return grad_input
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class Flatten(Layer):
    """
    Flatten a tensor into [batch_size, -1].
    """
    def __init__(self) -> None:
        super().__init__()
        self.input_shape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        assert self.input_shape is not None
        return grads.reshape(self.input_shape)

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.labels = None
        self.predicts = None
        self.probs = None
        self.grads = None

        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.predicts = predicts
        self.labels = labels.astype(np.int64)

        if self.has_softmax:
            self.probs = softmax(predicts)
        else:
            self.probs = predicts

        batch_indices = np.arange(self.labels.shape[0])
        picked_probs = self.probs[batch_indices, self.labels]
        picked_probs = np.clip(picked_probs, 1e-12, None)
        loss = -np.mean(np.log(picked_probs))
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        batch_size = self.labels.shape[0]
        batch_indices = np.arange(batch_size)

        if self.has_softmax:
            self.grads = self.probs.copy()
            self.grads[batch_indices, self.labels] -= 1
            self.grads /= batch_size
        else:
            self.grads = np.zeros_like(self.probs)
            selected = np.clip(self.probs[batch_indices, self.labels], 1e-12, None)
            self.grads[batch_indices, self.labels] = -1.0 / selected
            self.grads /= batch_size

        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition
