# Dear reader,
# when I wrote this code, only god and me knew what it does.
# Now only god knows.

import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10], cost=network.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.5, lmbda = 5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True,
        monitor_evaluation_cost=True, monitor_training_accuracy=True,  monitor_training_cost=True)
