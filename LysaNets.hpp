#ifndef AVLEARN_H
#define AVLEARN_H

#include "D:\rlv\Static_libraries\MatrixOperations/MatrixOperations.h"

using namespace std;
using namespace MatOp;

namespace LysaNets
{
extern Matrix one_hot(Matrix &Y, int n_classes);
extern double compute_cost(Matrix &A2, Matrix &Y);
extern double accuracy(Matrix pred, Matrix labels);
extern Matrix softmax(Matrix &z);
extern Matrix relu(Matrix &z);
extern Matrix relu_prime(Matrix &z);
extern Matrix leaky_relu(Matrix &z, double alpha);
extern Matrix leaky_relu_prime(Matrix &z, double alpha);

}//namespace end


#endif // AVLEARN_H
