#ifndef NNCLASSIFIER_HPP_INCLUDED
#define NNCLASSIFIER_HPP_INCLUDED

#include <vector>

#include "NeuralNetwork.hpp"

namespace LysaNets
{
class NNClassifier
{
    public:
        NNClassifier(NeuralNetwork *nn, int epochs);

        // functions
        void fit(Matrix &X, Matrix &Y);
        Matrix predict(Matrix &X, Matrix &Y);

        //getter
        vector<double> get_cost();
        NeuralNetwork get_model();

    private:
        int epochs = 1000;
        NeuralNetwork* model;
        vector<double> _cost;
};
}

#endif // NNCLASSIFIER_HPP_INCLUDED
