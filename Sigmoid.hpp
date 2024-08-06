#ifndef SIGMOID_HPP_INCLUDED
#define SIGMOID_HPP_INCLUDED

#include "D:\rlv\Static_libraries\MatrixOperations/MatrixOperations.h"

using namespace MatOp;

namespace LysaNets
{
class Sigmoid
{
    public:
        Matrix Sigm(Matrix &z);
        Matrix Prime(Matrix &z);

};
}

#endif // SIGMOID_HPP_INCLUDED
