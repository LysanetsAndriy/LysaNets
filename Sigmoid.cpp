#include "Sigmoid.hpp"

using namespace MatOp;

namespace LysaNets
{
Matrix Sigmoid::Sigm(Matrix &z)
{
    return 1.0 / (1.0 + (-z).exp());
}

Matrix Sigmoid::Prime(Matrix &z)
{
    return (Sigmoid::Sigm(z)) * (1.0 - Sigmoid::Sigm(z));
}
}
