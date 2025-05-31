import numpy as np
from structs.function import activation_functions
from numpy.lib.stride_tricks import as_strided

# FC layer
class fc_layer:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size, output_size) * 0.01 # random weight initialization
        self.bias = np.zeros(output_size) # random bias initialization
        self.activation = activation_functions[activation] # activation function
    
    def forward(self, x):
        # feed forward
        self.input = x # input
        self.z = np.dot(x, self.weights) + self.bias # z = Wx + b
        self.output = self.activation[0](self.z) # activation function
        return self.output
    
    def backward(self, grad_output, optimizer):
        # backpropagation
        if self.activation == activation_functions['softmax']:
            grad_z = grad_output  # activation이 softmax인 경우 CE_deriv의 결과(y_pred - y_true) 직접 전달
        else:
            grad_z = grad_output * self.activation[1](self.z)

        grad_weights = np.dot(self.input.T, grad_z) # gradient of weights
        grad_bias = np.sum(grad_z, axis=0)  # gradient of bias
        
        grad_input = np.dot(grad_z, self.weights.T) # gradient of input
    
        self.weights, self.bias = optimizer.step(
            self.weights, grad_weights, self.bias, grad_bias
        ) # update weights and bias

        return grad_input

# 2D max pooling layer
class max_pool2d_layer:
    def __init__(self, pool_size=2, stride=2, padding=0):
        # pool size가 정수인 경우 2차원 리스트로 변환
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = pool_size

        # stride가 정수인 경우 2차원 리스트로 변환
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        # padding가 정수인 경우 2차원 리스트로 변환
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

    def pad(self, x):
        # padding
        pad_h, pad_w = self.padding
        return np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')

    def unpad(self, x):
        # unpadding
        pad_h, pad_w = self.padding
        return x[:, pad_h:-pad_h, pad_w:-pad_w, :] if pad_h > 0 or pad_w > 0 else x

    def im2col(self, x, kh, kw, sh, sw):
        """
        입력 이미지 텐서를 슬라이딩 윈도우 방식으로 패치 단위로 분할하여
        (N, out_h, out_w, kh, kw, C) 형태의 텐서로 변환하는 함수.

        convolution 연산을 행렬 곱으로 처리할 수 있도록 함.

        Parameters:
            x  (ndarray): 입력 이미지 배열, shape (N, H, W, C)
            kh (int): 커널 높이 (kernel height)
            kw (int): 커널 너비 (kernel width)
            sh (int): 세로 방향 stride
            sw (int): 가로 방향 stride

        Returns:
            ndarray: 슬라이딩 윈도우로 추출된 패치, shape (N, out_h, out_w, kh, kw, C)
        """
        N, H, W, C = x.shape
        out_h = (H - kh) // sh + 1
        out_w = (W - kw) // sw + 1

        shape = (N, out_h, out_w, kh, kw, C)
        strides = (
            x.strides[0],
            x.strides[1] * sh,
            x.strides[2] * sw,
            x.strides[1],
            x.strides[2],
            x.strides[3]
        )
        return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    def forward(self, x):
        # feed forward
        self.x = x # input
        x_padded = self.pad(x)
        self.x_padded = x_padded # padded input

        kh, kw = self.pool_size # kernel size
        sh, sw = self.stride # stride

        patches = self.im2col(x_padded, kh, kw, sh, sw)  # shape: (N, out_h, out_w, kh, kw, C)
        self.patches = patches # patches

        patches_reshaped = patches.reshape(patches.shape[0], patches.shape[1], patches.shape[2], -1, patches.shape[-1])
        self.max_indices = np.argmax(patches_reshaped, axis=3) # max indices

        out = np.max(patches_reshaped, axis=3) # output
        return out

    def backward(self, grad_output):
        # backpropagation
        N, out_h, out_w, C = grad_output.shape # output shape
        kh, kw = self.pool_size # kernel size
        sh, sw = self.stride # stride
        H_p, W_p = self.x_padded.shape[1:3] # padded input shape

        grad_patches = np.zeros_like(self.patches.reshape(N, out_h, out_w, kh * kw, C)) # gradient of patches

        # (N, out_h, out_w, C) → (N, out_h, out_w, 1, C)
        flat_idx = self.max_indices[..., np.newaxis, :]
        # create mask to scatter gradients
        np.put_along_axis(grad_patches, flat_idx, grad_output[..., np.newaxis, :], axis=3)

        # reshape back to (N, out_h, out_w, kh, kw, C)
        grad_patches = grad_patches.reshape(N, out_h, out_w, kh, kw, C)

        # initialize gradient map
        grad_input_padded = np.zeros_like(self.x_padded) # gradient of padded grad_input

        # backpropagation
        for i in range(kh):
            for j in range(kw):
                grad_input_padded[:, i:H_p-kh+i+1:sh, j:W_p-kw+j+1:sw, :] += grad_patches[:, :, :, i, j, :]

        return self.unpad(grad_input_padded) # unpadding    

# Flatten layer
class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, x):
        # feed forward
        self.input_shape = x.shape # input shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output):
        # backpropagation
        return grad_output.reshape(self.input_shape)

# 2D convolution layer
class conv2d_layer:
    def __init__(self, input_size, output_size, kernel_size, stride, padding, activation):
        self.input_size = input_size # input size
        self.output_size = output_size # output size
        self.activation = activation_functions[activation] # activation function

        # kernel size가 정수인 경우 2차원 리스트로 변환
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        # stride가 정수인 경우 2차원 리스트로 변환
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        # padding가 정수인 경우 2차원 리스트로 변환
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        kh, kw = self.kernel_size
        self.weights = np.random.randn(output_size, kh, kw, input_size) * 0.01 # random weight initialization
        self.bias = np.zeros(output_size) # random bias initialization

    def pad(self, x):
        # padding
        pad_h, pad_w = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)
        return np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')

    def unpad(self, x):
        # unpadding
        pad_h, pad_w = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)
        return x[:, pad_h:-pad_h, pad_w:-pad_w, :]

    def im2col(self, x, kh, kw, sh, sw):
        """
        입력 이미지 텐서를 슬라이딩 윈도우 방식으로 패치 단위로 분할하여
        (N, out_h, out_w, kh, kw, C) 형태의 텐서로 변환하는 함수.

        convolution 연산을 행렬 곱으로 처리할 수 있도록 함.

        Parameters:
            x  (ndarray): 입력 이미지 배열, shape (N, H, W, C)
            kh (int): 커널 높이 (kernel height)
            kw (int): 커널 너비 (kernel width)
            sh (int): 세로 방향 stride
            sw (int): 가로 방향 stride

        Returns:
            ndarray: 슬라이딩 윈도우로 추출된 패치, shape (N, out_h, out_w, kh, kw, C)
        """
        N, H, W, C = x.shape # input shape
        out_h = (H - kh) // sh + 1 # output height
        out_w = (W - kw) // sw + 1 # output width
        shape = (N, out_h, out_w, kh, kw, C) # shape
        strides = (
            x.strides[0],
            x.strides[1] * sh,
            x.strides[2] * sw,
            x.strides[1],
            x.strides[2],
            x.strides[3]
        )
        return as_strided(x, shape=shape, strides=strides)

    def forward(self, x):
        # feed forward
        self.x = x # input
        x_padded = self.pad(x) # padding
        self.x_padded = x_padded # padded input
        kh, kw = self.kernel_size # kernel size
        sh, sw = self.stride # stride

        # convolution 연산을 행렬 곱으로 처리
        patches = self.im2col(x_padded, kh, kw, sh, sw)
        self.patches = patches
        out = np.einsum("nhwklc,oklc->nhwo", patches, self.weights) + self.bias
        self.z = out

        # activation function
        return self.activation[0](out)

    def backward(self, grad_output, optimizer):
        # backpropagation

        # 이전 layer가 flatten layer인 경우, reshape
        if grad_output.ndim != 4:
            grad_output = grad_output.reshape(self.z.shape)

        # activation function
        grad_z = grad_output * self.activation[1](self.z)

        # gradient of weights
        grad_weights = np.einsum("nhwklc,nhwo->oklc", self.patches, grad_z)
        grad_bias = np.sum(grad_z, axis=(0, 1, 2))

        kh, kw = self.kernel_size # kernel size
        sh, sw = self.stride # stride
        N, H_p, W_p, C_in = self.x_padded.shape # padded input shape
        grad_x_padded = np.zeros_like(self.x_padded) # gradient of padded input

        # backpropagation
        for i in range(kh):
            for j in range(kw):
                patch_grad = np.einsum("nhwo,oc->nhwc", grad_z, self.weights[:, i, j, :])
                grad_x_padded[:, i:H_p-kh+i+1:sh, j:W_p-kw+j+1:sw, :] += patch_grad
        
        # update weights and bias
        self.weights, self.bias = optimizer.step(self.weights, grad_weights, self.bias, grad_bias)
        
        # unpadding
        return self.unpad(grad_x_padded)
