#include <vector>
#include <iostream>
#include <iomanip>

#include "NNClassifier.hpp"
#include "NeuralNetwork.hpp"
#include "LysaNets.hpp"

using namespace std;

namespace LysaNets
{
NNClassifier::NNClassifier(NeuralNetwork *nn, int epochs)
{
    /*
        Arguments:
        model - instance of NeuralNetwork
        epochs (int) - number of epochs
    */

    this->model = nn;
    this->epochs = epochs;
    this->_cost;
}

vector<double> NNClassifier::get_cost()  { return _cost; }
NeuralNetwork NNClassifier::get_model() {return *(this->model);}

void NNClassifier::fit(Matrix &X, Matrix &Y)
{
    /*
        Learn weights and errors from training data

        Arguments:
        X - input data (number of features, number of examples)
        Y - labels (1, number of examples)
    */
    cout << '\n';
    for(int i = 0; i < this->epochs; i++)
    {
        Matrix *cache = (*this->model).forward_propagation(X);

        Matrix one_hot_matrix = LysaNets::one_hot(Y, (*this->model).get_n_classes());

        Matrix *grads = (*this->model).backward_propagation(X, one_hot_matrix , cache);

        (*this->model).update_parameters(grads);

        this->_cost.push_back(LysaNets::compute_cost(cache[3], one_hot_matrix));

        if (i == 330)
        {
            (*this->model).set_learning_rate(0.2);
        }
        if (i == 594)
        {
            (*this->model).set_learning_rate(0.1);
        }
        if (i == 759)
        {
            (*this->model).set_learning_rate(0.05);
        }
        if (i == 924)
        {
            (*this->model).set_learning_rate(0.01);
        }
        if (i == 1122)
        {
            (*this->model).set_learning_rate(0.001);
        }
        if (i == 1300)
        {
            (*this->model).set_learning_rate(0.0001);
        }
//        if (i == 60)
//        {
//            (*this->model).set_learning_rate((*this->model).get_learning_rate()*0.1);
//        }
//        if (i == 125)
//        {
//            (*this->model).set_learning_rate((*this->model).get_learning_rate()*0.1);
//        }
//        if (i == 1500)
//        {
//            (*this->model).set_learning_rate((*this->model).get_learning_rate()*0.1);
//        }
//        if (i == 750)
//        {
//            (*this->model).set_learning_rate((*this->model).get_learning_rate()*0.1);
//        }

        cerr << '\r' << "Current cost: " << setw( 8 ) << std::left << _cost.back() << " | ";
        cerr << "Training process: " << setw( 6 ) << std::right << fixed << setprecision(1) << i*(100.0/this->epochs) << '%'  << " | ";
    }
    cerr << '\r' << "Current cost: " << setw( 8 ) << std::left << _cost.back() << " | ";
    cerr << "Training process: " << setw( 6 ) << std::right << 100 << '%' << '\n';
}

Matrix NNClassifier::predict(Matrix &X, Matrix &Y)
{
    /*
        Generate array of predicted labels for the input dataset

        Arguments:
        X - input data (number of features, number of examples)

        Returns:
        predicted labels of shape (1, n_samples)
    */

    Matrix *cache = (*this->model).forward_propagation(X);

    Matrix one_hot_matrix = LysaNets::one_hot(Y, (*this->model).get_n_classes());
    cerr << LysaNets::compute_cost(cache[3], one_hot_matrix) << '\n';

    return cache[3].argmax(0);
}
}












