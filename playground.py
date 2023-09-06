
import math

from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(name, root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(name, format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})

    for n in nodes:
        dot.node(name=str(id(n)), label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

class Value:
    def __init__(self, data, _children=(), op='', label=''):
        self.data = data
        self.grad = 0.0
        self.label = label
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = op

    def __repr__(self):
        return f"Value({self.label}, data={self.data})"

    def __add__(self, other):
        # Make sure other is a Value type.
        other = other if isinstance(other, Value) else Value(other)
        # Do operation and set children.
        out = Value(self.data + other.data, (self, other), '+')
        def _backfunc():
            """dy/dx = 1.0 for addition operation.
            NOTE: For all _backfunc() we multiply by the out.grad for the chain rule.
            NOTE: For all _backfunc() we += the gradient in case a value is used more than once.
            """
            self.grad  += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backfunc
        return out

    def __radd__(self, other): # other + self
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        # Make sure other is a Value type.
        other = other if isinstance(other, Value) else Value(other)
        # Do operation and set children.
        out =  Value(self.data * other.data, (self, other), '*')
        def _backfunc():
            """
            out = self*other and need to find dOut/dSelf and dOut/dOther
            dOut/dSelf = other
            dOut/dOther = self
            """
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backfunc
        return out

    def __rmul__(self, other):  # other * self case
        return self * other

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "Only support int/float"
        out = Value(self.data**exponent, (self, ), f'**{exponent}')
        def _backfunc():
            """
            y = x^n
            dy/dx = n * x^(n-1) from wikipedia.
            """
            self.grad += exponent * (self.data**(exponent-1)) * out.grad
        out._backward = _backfunc
        return out

    def __truediv__(self, other):
        return self * other**-1

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        def _backfunc():
            """
            y = e^x
            dy/dx = x
            """
            self.grad = out.data * out.grad
        out._backward = _backfunc
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        def _backfunc():
            """
            y = tanh(x)
            dy/dx = 1 - tanh(x)^2 from wikipedia
            """
            self.grad += (1 - t**2) * out.grad
        out._backward = _backfunc
        return out

    def backward(self):
        """Run a full back propagation over all inputs computing gradients at each step.
        and storing that gradiant for every Value."""
        topo = []
        visited = set()
        def build_topo(v):
            """Build a recursive topographic map of all children."""
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0  # Reset gradient.
        for node in reversed(topo):
            node._backward()


a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a*b + c
print(d)

t = a / c
print(t)

simple_dot = draw_dot("simple_test", d, format='png')
simple_dot.render()


# Simple network test.
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# Weights
w1 = Value(-3.0, label='w1')
w2 = Value( 1.0, label='w2')
# Bias
b = Value(6.88137, label='b')
n = x1*w1 + x2*w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'
o.backward()

net_dot = draw_dot("simple_net", o, format='png')
net_dot.render()


# Simple network test 2.
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# Weights
w1 = Value(-3.0, label='w1')
w2 = Value( 1.0, label='w2')
# Bias
b = Value(6.88137, label='b')
n = x1*w1 + x2*w2 + b; n.label = 'n'
# spell out tanh(x)
temp = 2*n; temp.label='2n'
e = temp.exp()
o = (e - 1) / (e + 1)
o.label = 'o'
o.backward()

net_dot = draw_dot("simple_net2", o, format='png')
net_dot.render()

# Now do the same network with pytorch
import torch

x1 = torch.Tensor([ 2.0]).double(); x1.requires_grad = True
x2 = torch.Tensor([ 0.0]).double(); x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double(); w1.requires_grad = True
w2 = torch.Tensor([ 1.0]).double(); w2.requires_grad = True
b  = torch.Tensor([6.88137]).double();   b.requires_grad = True

n = x1*w1 + x2*w2 + b
o = torch.tanh(n)

print(o.data.item())
o.backward()

print("----")
print("pytorch")
print('x2', x2.grad.item())
print('w2', w2.grad.item())
print('x1', x1.grad.item())
print('w1', w1.grad.item())


# Build a neuron class like torch
import random
class Neuron():
    """Simple neuron class.  Single bias term, single weight for each input."""
    def __init__(self, numdim, nonlin=True):
        # Initialize the proper number of weights to random numbers.
        self.w = [Value(random.uniform(-1,1)) for _ in range(numdim)]
        # Assume starting with no bias.
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        """Take an input x, and calculate an output and return it."""
        # Multiply each input by the corrisponding weight, then add b.
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        if self.nonlin:
            out = act.tanh()  # Apply nonlinearity.
        return out

    def parameters(self):
        return self.w + [self.b]

# A layer is a collection of neurons.
class Layer:
    """Collection of neurons, one for each desired output."""
    def __init__(self, numin, numout):
        self.neurons = [Neuron(numin) for _ in range(numout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        # If we just have a single neuron, just return that single output.
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

# A multi-layer perceptron has several layers.
class MLP:
    def __init__(self, numin, layer_neuron_counts):
        # To make iterating easier, make one list.  The number of inputs
        # to the next layer is the same as the number of outputs of the
        # previous layer.
        size = [numin] + layer_neuron_counts
        num_layers = len(layer_neuron_counts)
        self.layers = [Layer(size[i], size[i+1]) for i in range(num_layers)]

    def __call__(self, x):
        for layer in self.layers:
            # The output of the layer is the new input for the next
            # layer, so we can just re-use x over and over.
            x = layer(x)
        return x  # Final output can return.

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

# To create the diagram of the MLP, it has 3 inputs, 4 neurons, then 4 neurons, then single output.
layer_neuron_counts = [4, 4, 1]
x = [2.0, 3.0, -1]  # 3 dimensional input.
mlp = MLP(3, layer_neuron_counts)
y = mlp(x)
print("Untrained model output: (expect 1.0)")
print(x, y)
print("num params: ", len(mlp.parameters()))


# Example data set with 4 examples.
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]  # Desired targets.

# Build a training loop.
for k in range(200):
    # Forward pass.
    ypred = [mlp(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # Backward pass.
    for p in mlp.parameters():
        p.grad = 0.0  # Reset grads before the pass!!
    loss.backward()

    # Update the weights and biases.
    for p in mlp.parameters():
        p.data += -0.01 * p.grad

    if not (k % 10):
        print(k, loss.data)

# Forward pass and check.
ypred = [mlp(x) for x in xs]
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
print("Loss: ", loss.data)

print(ypred)
