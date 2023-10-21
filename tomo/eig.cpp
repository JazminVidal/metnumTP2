// 1.1.

#include <stdlib.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <eigen3/Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

VectorXd power_iteration_eigenvector(const MatrixXd A, const int niter, const double eps) {
    auto const rows = A.rows();
    VectorXd v = VectorXd::Random(rows);
    // v /= v.norm();
    for (int i = 0; i < niter; i++) {
        VectorXd prev_v = v;
        v = A*v;
        v /= v.norm();
        if (abs(v.transpose()*prev_v - 1.0) < eps) {
            break;
        }
    }
    return v;
}

double power_iteration_eigenvalue(const MatrixXd A, const VectorXd v) {
    double n2 = v.norm()*v.norm();
    double eval = v.transpose()*A*v;
    eval /= n2;
    return eval;
}

MatrixXd eig(const MatrixXd A, const int num, const int niter, const double eps) {
    auto const rows = A.rows();
    MatrixXd res(rows + 1, rows);
    MatrixXd M = A;
    VectorXd evals(num);
    for (int i = 0; i < num; i++) {
        VectorXd v = power_iteration_eigenvector(M, niter, eps);
        double eval = power_iteration_eigenvalue(M, v);
        res.row(i) << v.transpose();
        evals[i] = eval;
        M = M - eval*(v*v.transpose());
    }
    res.row(rows) << evals.transpose();
    return res;
}

// VectorXd matrix_vector_multiplication(const MatrixXd& matrix, const VectorXd& vector) {
//     return matrix * vector;
// }

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " input_file output_eigenvectors_file output_eigenvalues_file num niter eps" << std::endl;
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_eigenvectors_file = argv[2];
    const char* output_eigenvalues_file = argv[3];
    const int num = atoi(argv[4]);
    const int niter = atoi(argv[5]);
    const double eps = atof(argv[6]);

    std::ifstream fin(input_file);
    if (!fin.is_open()) {
        std::cerr << "Error: could not open input file " << input_file << std::endl;
        return 1;
    }

    // Read matrix from file
    int nrows, ncols;
    fin >> nrows >> ncols;
    MatrixXd A(nrows, ncols);
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            fin >> A(i, j);
        }
    }
    fin.close();

    // Perform operation
    MatrixXd res = eig(A, num, niter, eps);

    // Write results to output files
    std::ofstream fout1(output_eigenvectors_file);
    if (!fout1.is_open()) {
        std::cerr << "Error: could not open output file " << output_eigenvectors_file << std::endl;
        return 1;
    }
    for (int i = 0; i < nrows; i++) {
        fout1 << res.col(i).head(nrows).transpose() << std::endl;
    }
    fout1.close();

    std::ofstream fout2(output_eigenvalues_file);
    if (!fout2.is_open()) {
        std::cerr << "Error: could not open output file " << output_eigenvalues_file << std::endl;
        return 1;
    }
    fout2 << res.row(nrows) << std::endl;
    fout2.close();
    return 0;
}