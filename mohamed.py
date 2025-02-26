import random


def tanh(x):
    return (1 - 2 / (1 + 2 ** (2 * x)))


def dot_product(A, S):
    return sum(x * y for x, y in zip(A, S))

def neural_network(inputs, weights1, weights2, A, S):
    hidden_layer_output = [tanh(dot_product(inputs, w) + b1) for w in weights1]

    output = tanh(dot_product(hidden_layer_output, weights2) + b2)

    return output

inputs = [1.0, 0.5, -0.5]
weights1 = [[random.uniform(-0.5, 0.5) for _ in range(4)] for _ in range(3)]
weights2 = [random.uniform(-0.5, 0.5) for _ in range(5)]
b1 = random.uniform(-0.5, 0.5)
b2 = random.uniform(-0.5, 0.5)

output = neural_network(inputs, weights1, weights2, b1, b2)

print("Network:", output)
