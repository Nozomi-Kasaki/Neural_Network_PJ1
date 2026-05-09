from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(
        self,
        input_shape=(1, 28, 28),
        num_classes=10,
        conv_channels=(16, 32, 64),
        kernel_size=3,
        conv_strides=(1, 2, 2),
        padding=1,
        fc_hidden_dim=128,
        lambda_list=None
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.conv_strides = conv_strides
        self.padding = padding
        self.fc_hidden_dim = fc_hidden_dim
        self.lambda_list = lambda_list
        self.layers = []

        if input_shape is not None:
            self._build_layers()

    def _he_initializer(self, fan_in):
        std = np.sqrt(2.0 / fan_in)
        return lambda size: np.random.normal(loc=0.0, scale=std, size=size)

    def _apply_weight_decay(self, layer, decay_idx):
        if self.lambda_list is not None and decay_idx < len(self.lambda_list):
            layer.weight_decay = True
            layer.weight_decay_lambda = self.lambda_list[decay_idx]

    def _build_layers(self):
        in_channels, cur_h, cur_w = self.input_shape
        self.layers = []
        decay_idx = 0

        for out_channels, stride in zip(self.conv_channels, self.conv_strides):
            fan_in = in_channels * self.kernel_size * self.kernel_size
            conv_layer = conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=stride,
                padding=self.padding,
                initialize_method=self._he_initializer(fan_in)
            )
            self._apply_weight_decay(conv_layer, decay_idx)
            self.layers.append(conv_layer)
            self.layers.append(ReLU())

            cur_h = (cur_h + 2 * self.padding - self.kernel_size) // stride + 1
            cur_w = (cur_w + 2 * self.padding - self.kernel_size) // stride + 1
            in_channels = out_channels
            decay_idx += 1

        self.layers.append(Flatten())

        linear_in_dim = in_channels * cur_h * cur_w
        hidden_layer = Linear(
            in_dim=linear_in_dim,
            out_dim=self.fc_hidden_dim,
            initialize_method=self._he_initializer(linear_in_dim)
        )
        self._apply_weight_decay(hidden_layer, decay_idx)
        self.layers.append(hidden_layer)
        self.layers.append(ReLU())
        decay_idx += 1

        output_layer = Linear(
            in_dim=self.fc_hidden_dim,
            out_dim=self.num_classes,
            initialize_method=self._he_initializer(self.fc_hidden_dim)
        )
        self._apply_weight_decay(output_layer, decay_idx)
        self.layers.append(output_layer)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            state = pickle.load(f)

        self.input_shape = tuple(state['input_shape'])
        self.num_classes = state['num_classes']
        self.conv_channels = tuple(state['conv_channels'])
        self.kernel_size = state['kernel_size']
        self.conv_strides = tuple(state['conv_strides'])
        self.padding = state['padding']
        self.fc_hidden_dim = state['fc_hidden_dim']
        self.lambda_list = state.get('lambda_list')

        self._build_layers()

        optimizable_layers = [layer for layer in self.layers if layer.optimizable]
        for layer, params in zip(optimizable_layers, state['params']):
            layer.W = params['W']
            layer.b = params['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = params['weight_decay']
            layer.weight_decay_lambda = params['lambda']
        
    def save_model(self, save_path):
        param_list = []
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })

        state = {
            'model_type': 'Model_CNN',
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'conv_channels': self.conv_channels,
            'kernel_size': self.kernel_size,
            'conv_strides': self.conv_strides,
            'padding': self.padding,
            'fc_hidden_dim': self.fc_hidden_dim,
            'lambda_list': self.lambda_list,
            'params': param_list
        }

        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
