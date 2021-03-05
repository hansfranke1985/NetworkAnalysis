from functools import reduce

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt 


class Conv2D:
    '''A Convolution layer using nxn filters.
    
    A simple function that achieves the convolution operation efficiently for two-dimensional inputs and three-dimensional inputs. 
    a set of convolutional filters (‘kernels’ in Keras’s terminology)
    an input layer (or image) as inputs. 

    The input layer should have a third dimension or two dimension, 
    representing a stack of feature maps, and each filter should have a third dimension of corresponding size. 

    The function should output a number of two-dimensional feature maps corresponding to the number of input filters, 
    though these can be stacked into a third dimensional like the input layer. 
            
    TODO: 3d
    TODO: padding
    '''

    def __init__(self, num_filters, kernal_size):
        '''
            filters is a 3 dimensions array (num_filters, 3, 3)
        '''
        self.num_filters = num_filters
        self.kernal_size = kernal_size
        self.filters = np.random.randn(num_filters, 3, 3)
        
    
    
    def iterate_regions(self, image):
        '''Generates image regions    
        ''' 
        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + self.kernal_size), j:(j + self.kernal_size)]
                yield im_region, i, j
                

    def sub_forward(self, inputs):
        '''Return a 3 dimensions array
            
        ::inputs: 28x28
        ::outputs: 26x26x8
        '''
        # (28, 28)
        h, w = inputs.shape

        # for now, padding = 0 and stride = 1 
        outputs = np.zeros((h - self.kernal_size + 1, w - self.kernal_size + 1, self.num_filters))
    
        for im_region, i, j in self.iterate_regions(inputs):
            outputs[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        return outputs   
    
    def forward(self, inputs):
        # cache input for backpro
        # self.last_inputs = inputs
        
        if len(inputs.shape) == 2:
            return self.sub_forward(inputs)
        
        elif len(inputs.shape) == 3:
            permuted = np.transpose(inputs, (2, 0, 1))
            c, h, w = permuted.shape
            
            container = np.zeros((h - self.kernal_size + 1, w - self.kernal_size + 1, self.num_filters))

            for i in range(c):
                outputs = self.sub_forward(permuted[i])
                container += outputs
            return container     
        else:
            raise AttributeError
            
class Activation:
    '''Activation function Implement
    '''
    def __init__(self):
        pass

    def relu(self, in_features):
        '''A simple function that achieves rectified linear (relu) activation over a whole feature map, with a threshold at zero. 

        in_features can be numpy array, scalar, vector, or matrix
        '''
        return np.maximum(0, in_features)

    def sigmoid(self, in_features):
        '''Apply sigmoid activation function
        
        in_features can be numpy array, scalar, vector, or matrix
        '''
        return 1/(1+np.exp(-in_features))
    
    def leakyRelu(self, in_features, alpha=0.1):
        '''Apply leakyRelu activation function
        
        in_features can be numpy array, scalar, vector, or matrix
        '''
        return np.where(in_features > 0, in_features, in_features * alpha)      
    
    def softmax(self, in_features):
        '''A function that converts the activation of a 1-dimensional matrix (such as the output of a fully-connected layer) 
        into a set of probabilities that each matrix element is the most likely classification. 

        This should include the algorithmic expression of a softmax (normalised exponential) function.
        
        in_features can be numpy array, scalar, vector, or matrix
        '''
        expo = np.exp(in_features)
        expo_sum = np.sum(expo)
        return expo/expo_sum

class MaxPooling:
    '''Specify the spatial extent of the pooling, with the size of the output feature map changing accordingly
    '''
    def __init__(self, pool=2, stride=2):
        self.pool = pool 
        self.stride = stride 

    def iterate_regions(self, image):
        '''Generates non-overlapping kxk image regions to pool over
        '''
        h, w, c = image.shape
        
        # floor() the value
        new_h = int(np.floor(h/self.pool))
        new_w = int(np.floor(w/self.pool))
                
        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * self.pool):(i * self.pool + self.stride), (j * self.pool):(j * self.pool + self.stride)]
                yield im_region, i, j

    def forward(self, inputs):
        '''Apply a forward for the maxpooling layer
        
        ::output is a 3d numpy array with dimensions (floor(h/2), floor(w/2), num_filters).
        ::input is a 3d numpy array with dimensions (h, w, num_filters)
        '''
        h, w, num_filters = inputs.shape
        
        # floor() the value
        new_h = int(np.floor(h/self.pool))
        new_w = int(np.floor(w/self.pool))
        
        output = np.zeros((new_h, new_w, num_filters))

        for im_region, i, j in self.iterate_regions(inputs):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output


class Normalization():
    '''Normalization Implement
    
    TODO: for all filter or overall?
    '''
    def __init__(slef):
        self.epsilon = np.finfo(float).eps
        # self.epsilon=1e-10
        
    
    def zeromean(self,in_features):
        '''Normalisation within each feature map, modifying the feature map 
        so that its mean value is zero and its standard deviation is one.
        '''
        return (in_features - np.mean(in_features, axis=0))/ ( np.std(in_features, axis=0)+ self.epsilon )
    
    def minmax(self,in_features):
        '''min-max normalization
        '''
        return (in_features - np.amin(in_features, axis=0)) / (np.amax(in_features, axis=0)-np.amin(in_features, axis=0) + self.epsilon)
    
    def loge(self,in_features):
        '''log transform normalization
        
        note: np.log is ln, whereas np.log10 is standard base 10 log.
        '''
        return np.log(in_features)/np.log(np.amax(in_features, axis=0))
    
    def log10(self,in_features):
        '''log transform normalization
        
        note: np.log is ln, whereas np.log10 is standard base 10 log.
        '''
        return np.log10(in_features)/np.log10(np.amax(in_features, axis=0))
    
    
class FC:
    '''fully-connected layer
    specify the number of output nodes, and link each of these to every node a stack of feature maps. 
    the stack of feature maps will typically be flattened into a 1-dimensional matrix first. 
    '''
#     def __init__(self, in_dim, out_dim):
#         '''Divide by in_dim to reduce the variance of our initial values
        
#         in_dim = Inputs Numbers of Neuron
#         out_dim = Outputs Numbers of Neuron
#         '''
#         self.weights = np.random.randn(in_dim, out_dim) / in_dim
#         self.biases = np.zeros(out_dim)

    def __init__(self):
        '''Divide by in_dim to reduce the variance of our initial values
        
        in_dim = Inputs Numbers of Neuron
        out_dim = Outputs Numbers of Neuron
        '''
        self.weights = np.zeros(0)
        self.biases = np.zeros(0)


    def forward(self, inputs, out_dim):
        '''Returns a 1d numpy array
        '''
        if not self.weights.any():
            in_dim = reduce(lambda x,y:x*y,inputs.shape)
            self.weights = np.random.randn(in_dim, out_dim) / in_dim
            self.biases = np.zeros(out_dim)
        
        inputs = inputs.flatten()

        in_dim, out_dim = self.weights.shape

        
        outputs = np.dot(inputs, self.weights) + self.biases

        return outputs
    
    def backforward(self, out, learning_rate=0.1):
        matrix = np.dot(self.weights, out)
        self.weights = -learning_rate * matrix
        return self.weights

class Softmax:
    '''A standard fully-connected layer with softmax activation.
    
    Refer https://deepai.org/machine-learning-glossary-and-terms/softmax-layer
    '''

    def __init__(self, in_dim, out_dim):
        '''Divide by in_dim to reduce the variance of our initial values
        
        in_dim = Inputs Numbers of Neuron
        out_dim = Outputs Numbers of Neuron
        '''

        self.weights = np.random.randn(in_dim, out_dim) / in_dim
        self.biases = np.zeros(out_dim)

    def __init__(self):
        '''Divide by in_dim to reduce the variance of our initial values
        
        in_dim = Inputs Numbers of Neuron
        out_dim = Outputs Numbers of Neuron
        '''
        self.weights = np.zeros(0)
        self.biases = np.zeros(0)


    def forward(self, inputs, out_dim):
        '''
        Performs a forward pass of the softmax layer using the given input.
        Returns a 1d numpy array containing the respective probability values.
        - input can be any array with any dimensions.
        '''
        if not self.weights.any():
            in_dim = reduce(lambda x,y:x*y,inputs.shape)
            self.weights = np.random.randn(in_dim, out_dim) / in_dim
            self.biases = np.zeros(out_dim)
            
        self.last_input_shape = inputs.shape
        
        inputs = inputs.flatten()
        self.last_input = inputs

        in_dim, out_dim = self.weights.shape

        
        feature = np.dot(inputs, self.weights) + self.biases
        self.last_feature = feature

        # softmax function
        expo = np.exp(feature)
        expo_sum = np.sum(expo, axis=0)
        out = expo / expo_sum
        
        
        
        # The input’s shape before we flatten it.
        #self.last_input_shape = input.shape
        # The input after we flatten it.
        #self.last_input = input
        # The totals, which are the values passed in to the softmax activation.
        #self.last_totals = totals
        
        return out
    
#     def backforward(self, out, learning_rate=0.1):
#         self.weights = -learning_rate *(self.weights*out)
#         return self.weights
    def backforward(self, L_out, learning_rate):

        for i, gradient in enumerate(L_out):
            if gradient == 0:
                continue

            # e totals
            exp = np.exp(self.last_feature)

            # Sum of all e totals
            sum_exp = np.sum(exp)


            # Gradients of totals against weights/biases/input
            z_b = 1
            z_w = self.last_input
            
            
            # Gradients of out[i] against totals, set the i!=k value for all and update i==k
            # if i!=k
            out_z = -exp[i] * exp / (sum_exp * sum_exp)
            # if i==k
            out_z[i] = exp[i] * (sum_exp - exp[i]) / (sum_exp * sum_exp)
            # 2-dimention, 1000*10
            z_inputs = self.weights

            # Gradients of loss against totals
            # (10,)
            L_z = gradient * out_z

            # Gradients for weights, biases and input
            L_b = L_z * z_b
            # 1000*1 x 1*10 => 1000*10
            L_w = np.dot(z_w[np.newaxis].T, L_z[np.newaxis] )
            # 1000*10 x 10*1 => 1000*1
            L_inputs = np.dot(z_inputs, L_z)

            # Update weights / biases
            self.weights = self.weights - learning_rate * L_w
            self.biases = self.biases - learning_rate * L_b
            # 1000 -> (h,w,c) = (13,13,8)
            out = L_inputs.reshape(self.last_input_shape)
            
            # print(self.biases)
            # print(self.weights)
            return out

class clac:
    '''write some utility funtion 
    '''
    def count_dimension(inputs):
        '''count dimention
        '''
        return reduce(lambda x,y:x*y,inputs.shape)
        
    