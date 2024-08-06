#include <fstream>
#include <cmath>

#include "NeuralNetwork.hpp"
#include "LysaNets.hpp"

namespace LysaNets
{
NeuralNetwork::NeuralNetwork()
{
}

NeuralNetwork::NeuralNetwork(int n_features, int n_hidden_units,
                             int n_classes, double learning_rate)
{
    /*
        Arguments:
        n_features: (int) number of features
        n_hidden_units: (int) number of hidden units
        n_classes: (int) number of classes
        learning_rate: (double)
    */

    //reg=Regularization(0.1, 0.2);
    //this->reg = reg
    this->n_features = n_features;
    this->n_classes = n_classes;
    this->learning_rate = learning_rate;
    this->n_hidden_units = n_hidden_units;
    this->sigm=Sigmoid();

    initialize_parameters();
}

Matrix NeuralNetwork::get_W1() {return W1;}
Matrix NeuralNetwork::get_b1() {return b1;}
Matrix NeuralNetwork::get_W2() {return W2;}
Matrix NeuralNetwork::get_b2() {return b2;}
int NeuralNetwork::get_n_features() {return n_features;}
int NeuralNetwork::get_n_classes() {return n_classes;}
int NeuralNetwork::get_n_hidden_units() {return n_hidden_units;}
double NeuralNetwork::get_learning_rate() {return learning_rate;}

void NeuralNetwork::set_learning_rate(double lr)
{
    this->learning_rate = lr;
}

Matrix* NeuralNetwork::get_parameters()
{
    /*
        Returns:
        Array with Matrix type parameters (parameters[0] = W1, parameters[1] = b1, parameters[2] = W2, parameters[3] = b2)
    */

    Matrix *parameters = new Matrix[4];
    parameters[0] = this->W1;
    parameters[1] = this->b1;
    parameters[2] = this->W2;
    parameters[3] = this->b2;

    return parameters;
}

void NeuralNetwork::store_parameters(std::string fname)
{
    std::ofstream fout(fname);

    for(int i = 0; i < this->n_hidden_units; i++)
    {
        for(int j = 0; j < this->n_features; j++)
        {
            fout << this->W1.getElement(i,j) << ' ';
        }
        fout << '\n';
    }

    for(int i = 0; i < this->n_hidden_units; i++)
    {
        fout << this->b1.getElement(i,0);
        fout << '\n';
    }

    for(int i = 0; i < this->n_classes; i++)
    {
        for(int j = 0; j < this->n_hidden_units; j++)
        {
            fout << this->W2.getElement(i,j) << ' ';
        }
        fout << '\n';
    }

    for(int i = 0; i < this->n_classes; i++)
    {
        fout << this->b2.getElement(i,0);
        fout << '\n';
    }
}

void NeuralNetwork::initialize_parameters()
{
    /*
        W1 - weight matrix (n_hidden_units, n_features)
        b1 - bias vector (n_hidden_units, 1)
        W2 - weight matrix (n_classes, n_hidden_units)
        b2 - bias vector (n_classes, 1)
    */

    this->W1 = Matrix::norm(0, 1.5, this->n_hidden_units, this->n_features);
    this->b1 = Matrix::zeros(this->n_hidden_units, 1);
    this->W2 = Matrix::norm(0, 1.5, this->n_classes, this->n_hidden_units);
    this->b2 = Matrix::zeros(this->n_classes, 1);
}

Matrix* NeuralNetwork::forward_propagation(Matrix &X)
{
    /*
        Arguments:
        X - input data (number of features, number of examples)

        Returns:
        Array of Matrix type a (a[0] = Z1, a[1] = A1, a[2] = Z2, a[3] = A2, a[0] = Z3, a[1] = A3)
    */

    Matrix * a = new Matrix[4];
    a[0] = (this->W1^X) + this->b1;   // Z1
    a[1] = leaky_relu(a[0], 0.01);   // A1
    a[2] = (this->W2^a[1]) + this->b2;// Z2
    a[3] = this->sigm.Sigm(a[2]);   // A2

    return a;

}

Matrix* NeuralNetwork::backward_propagation(Matrix &X, Matrix &Y, Matrix *cache)
{
    /*
        Arguments:
        X - input data (number of features, number of examples)
        Y - one-hot encoded vector of labels (n_classes, n_samples)
        cache - Array of Matrix type (cache[0] = Z1, cache[1] = A1, cache[2] = Z2,
                                      cache[3] = A2, cache[4] = Z3, cache[5] = A3)

        Returns:
        Array of type Matrix with gradients (grads[0] = dW1, grads[1] = db1, grads[2] = dW2,
                                             grads[3] = db2, grads[4] = dW3, grads[5] = db3)
    */
    int m = X.getColumn();

    Matrix A1 = cache[1];
    Matrix A2 = cache[3];

//    Matrix L1 = reg.l1_grad(W1, W2, m); add Regularization class
//    Matrix L2 = reg.l2_grad(W1, W2, m);

    Matrix dA2 = - (Y / A2 - (1 - Y)/(1 - A2));

//    Matrix dZ2 = A2 - Y;
    Matrix dZ2 = dA2 * A2 * (1 - A2);

    Matrix dW2 = (1.0 / m) * (dZ2 ^ (A1.T())); // + L1[0] + L2[0];
    Matrix db2 = (1.0 / m) * dZ2.sum(1).T();

    Matrix dZ1 = ((this->W2.T()) ^ dZ2) * leaky_relu_prime(A1, 0.01);
//    Matrix dZ1 = ((this->W2.T()) ^ dZ2) * (1 - A1 * A1);

    Matrix dW1 = (1.0 / m) * (dZ1 ^ (X.T())); // + L1[0] + L2[0];
    Matrix db1 = (1.0 / m) * dZ1.sum(1).T();

    Matrix *grads = new Matrix[4];
    grads[0] = dW1;
    grads[1] = db1;
    grads[2] = dW2;
    grads[3] = db2;

    return grads;
}

void NeuralNetwork::update_parameters(Matrix *grads)
{
    /*
        Update parameters with gradient descent

        Arguments:
        grads - Array of type Matrix with gradients (grads[0] = dW1, grads[1] = db1, grads[2] = dW2,
                                                     grads[3] = db2, grads[4] = dW3, grads[5] = db3)
    */

    Matrix dW1 = grads[0];
    Matrix db1 = grads[1];
    Matrix dW2 = grads[2];
    Matrix db2 = grads[3];

    this->W1 = this->W1 - this->learning_rate * dW1;
    this->b1 = this->b1 - this->learning_rate * db1;
    this->W2 = this->W2 - this->learning_rate * dW2;
    this->b2 = this->b2 - this->learning_rate * db2;
}

}















