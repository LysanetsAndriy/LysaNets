#include "LysaNets.hpp"

#include "D:\rlv\Static_libraries\MatrixOperations\MatrixOperations.h"

using namespace std;
using namespace MatOp;

namespace LysaNets
{
Matrix one_hot(Matrix &Y, int n_classes)
{
    /*
        Y - input labels of shape (1, n_samples)
        n_classes - number of classes
    */

    Matrix B = Matrix::zeros(n_classes, Y.getColumn());
    double **r = B.getMatrix();
    for (int i = 0; i < Y.getColumn(); i++)
        r[(int)Y.getElement(0,i)][i] = 1.0;
    Matrix res(n_classes, Y.getColumn());
    res.setM(r);
    return res;
}


double compute_cost(Matrix &A2, Matrix &Y)
{
    /*
        A2 - sigmoid output of the hidden layer activation, of shape (classes, n_examples)
        Y - labels of shape (classes, n_examples)
    */

    double m = Y.getRow();

    double cost;
    cost = (-1 / m)*((Y * A2.log()).sum() + ((1 - Y) * (1 - A2).log()).sum());
    return cost;
}

double accuracy(Matrix pred, Matrix labels)
{
    /*
        pred - predicted labels of shape (1, n_samples)
        labels - input labels of shape (1, n_samples)
    */

    int n_examples = labels.getColumn();
    double **p = pred.getMatrix();
    double **l = labels.getMatrix();
    int sm = 0;
    for(int i = 0; i < n_examples; i++)
    {
        if (p[0][i] == l[0][i])
            sm++;
    }
    return (1.0 * sm) / (1.0 * n_examples);
}

Matrix softmax(Matrix &z)
{
    Matrix exp_z = z.exp();
    return exp_z / exp_z.sum();
}

Matrix relu(Matrix &z)
{
    double **tmp = new double*[z.getRow()];
    for(int i = 0; i < z.getRow(); i++)
    {
        tmp[i] = new double[z.getColumn()];
        for (int j = 0; j < z.getColumn(); j++)
        {
            tmp[i][j] = max(z.getElement(i,j), 0.0);
        }
    }
    Matrix mx_z(z.getRow(), z.getColumn());
    mx_z.setM(tmp);
    return mx_z;
}

Matrix relu_prime(Matrix &z)
{
    double **tmp = new double*[z.getRow()];
    for(int i = 0; i < z.getRow(); i++)
    {
        tmp[i] = new double[z.getColumn()];
        for (int j = 0; j < z.getColumn(); j++)
        {
            if (z.getElement(i,j) > 0)
                tmp[i][j] = 1;
            else
                tmp[i][j] = 0;
        }
    }
    Matrix mx_z(z.getRow(), z.getColumn());
    mx_z.setM(tmp);
    return mx_z;
}

Matrix leaky_relu(Matrix &z, double alpha)
{
    double **tmp = new double*[z.getRow()];
    for(int i = 0; i < z.getRow(); i++)
    {
        tmp[i] = new double[z.getColumn()];
        for (int j = 0; j < z.getColumn(); j++)
        {
            tmp[i][j] = max(z.getElement(i,j), z.getElement(i,j)*alpha);
        }
    }
    Matrix mx_z(z.getRow(), z.getColumn());
    mx_z.setM(tmp);
    return mx_z;
}

Matrix leaky_relu_prime(Matrix &z, double alpha)
{
    double **tmp = new double*[z.getRow()];
    for(int i = 0; i < z.getRow(); i++)
    {
        tmp[i] = new double[z.getColumn()];
        for (int j = 0; j < z.getColumn(); j++)
        {
            if (z.getElement(i,j) > 0)
                tmp[i][j] = 1;
            else
                tmp[i][j] = alpha;
        }
    }
    Matrix mx_z(z.getRow(), z.getColumn());
    mx_z.setM(tmp);
    return mx_z;
}

}









