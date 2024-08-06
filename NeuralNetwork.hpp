#ifndef NEURALNETWORK_HPP_INCLUDED
#define NEURALNETWORK_HPP_INCLUDED

#include <iostream>

#include "Sigmoid.hpp"
#include "D:\rlv\Static_libraries\MatrixOperations/MatrixOperations.h"

using namespace MatOp;
using namespace std;

namespace LysaNets
{
class NeuralNetwork
{
    public:
        NeuralNetwork();
        NeuralNetwork(int n_features, int n_hidden_units, int n_classes,
                      double learning_rate);

        // functions
        Matrix* forward_propagation(Matrix &X);
        Matrix* backward_propagation(Matrix &X, Matrix &Y, Matrix *cache);
        void update_parameters(Matrix *grads);

        // Getters
        Matrix get_W1();
        Matrix get_b1();
        Matrix get_W2();
        Matrix get_b2();

        int get_n_features();
        int get_n_classes();
        int get_n_hidden_units();
        double get_learning_rate();
        void set_learning_rate(double lr);
        Matrix* get_parameters();
        void store_parameters(string fname);

    private:
        int n_features;
        int n_classes;
        int n_hidden_units;
        double learning_rate;
        //Regularization reg;
        Sigmoid sigm;
        Matrix W1;
        Matrix b1;
        Matrix W2;
        Matrix b2;

        void initialize_parameters();


};
}

#endif // NEURALNETWORK_HPP_INCLUDED
