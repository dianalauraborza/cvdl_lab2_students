import math
import torch
import numpy as np


class SoftmaxClassifier:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.W = None
        self.initialize()

    def initialize(self):
        # TODO your code here
        # initialize the weight matrix (remember the bias trick) with small random variables
        # you might find torch.randn userful here *0.001
        self.W = None
        # don't forget to set call requires_grad_() on the weight matrix,
        # as we will be be taking its gradients during the learning process
        pass

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        # TODO your code here
        # 0. compute the dot product between the weight matrix and the input X
        # you can use @ for this operation
        scores = None
        # remember about the bias trick!
        # 1. apply the softmax function on the scores, see torch.nn.functional.softmax
        # think about on what dimension (dim parameter) you should apply this operation
        scores = None
        # 2, returned the normalized scores
        return scores

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        # TODO your code here
        # 0. compute the dot product between the weight matrix and the input X as the scores
        scores = None
        # 1. compute the prediction by taking the argmax of the class scores
        # you might find torch.argmax useful here.
        # think about on what dimension (dim parameter) you should apply this operation
        label = None
        return label

    def cross_entropy_loss(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # TODO your code here
        loss = None
        return None

    def log_softmax(self, x: torch.Tensor) -> torch.Tensor:
        return None

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor,
            **kwargs) -> dict:

        history = []

        bs = kwargs['bs'] if 'bs' in kwargs else 128
        reg_strength = kwargs['reg_strength'] if 'reg_strength' in kwargs else 1e3
        epochs = kwargs['epochs'] if 'epochs' in kwargs else 100
        lr = kwargs['lr'] if 'lr' in kwargs else 1e-3
        print('hyperparameters: lr{:.4f}, reg{:.4f}, epochs{:.2f}'.format(lr, reg_strength, epochs))

        for epoch in range(epochs):
            print(epoch)
            for ii in range((X_train.shape[0] - 1) // bs + 1):  # in batches of size bs
                # TODO your code here
                start_idx = None  # we are ii batches in, each of size bs
                end_idx = None  # get bs examples

                # get the training training examples xb, and their coresponding annotations
                xb = X_train[start_idx:end_idx]
                yb = y_train[start_idx:end_idx]

                # apply the linear layer on the training examples from the current batch
                pred = None
                pred = None

                # compute the loss function
                # also add the L2 regularization loss (the sum of the squared weights)
                loss = None
                history.append(loss.detach().numpy())

                # start backpropagation: calculate the gradients with a backwards pass
                loss.backward()

                # update the parameters
                with torch.no_grad():  # we don't want to track gradients
                    # take a step in the negative direction of the gradient, the learning rate defines the step size
                    self.W -= self.W.grad * lr

                    # ATTENTION: you need to explictly set the gradients to 0 (let pytorch know that you are done with them).
                    self.W.grad.zero_()

        return history

    def get_weights(self, img_shape) -> np.ndarray:
        # TODO your code here
        W = self.W.detach().numpy()
        # 0. ignore the bias term
        W = None
        # 1. reshape the weights to (*image_shape, num_classes)
        W = None
        # you might find the transpose function useful here
        W = None
        return W

    def load(self, path: str) -> bool:
        # TODO your code here
        # load the input shape, the number of classes and the weight matrix from a file
        # you might find torch.load useful here
        self.W = None
        # don't forget to set the input_shape and num_classes fields
        self.num_classes = None
        self.input_shape = None
        return True

    def save(self, path: str) -> bool:
        # TODO your code here
        # save the input shape, the number of classes and the weight matrix to a file
        # you might find torch useful for this
        # TODO your code here
        return True

