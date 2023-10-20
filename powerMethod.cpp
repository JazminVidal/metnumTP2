/////////
// The power method is an iterative technique for determining the dominant eigenvalue and eigenvector of a matrix. This is the eigenvalue with
// the largest absolute value. The eigenvector is the one associated with the eigenvalue. 
// ASUME: 
// 1. A is a square matrix with n rows and n columns.
// 2. A has n eigenvalues
// 3. A has n linearly independent eigenvectors.
// 4. One eigenvalue is larger in magnitude. 
// Como los autovectores son LI y forman una base, eso nos permite decir que existen Betas tales que podemos escribir cualquier vector x 
// como una combinación lineal de ellos. Acá vienen las cuentas y en el infinito...
//


#include <fstream>
#include <iostream>
#include <utility>
#include <eigen3/Eigen/Dense>

using namespace Eigen;
using namespace std;

pair<MatrixXd, bool> dataloader(const char* filename) {
    ifstream fin(filename);
    if (!fin.is_open()) {
        cerr << "Error: could not open input file " << filename << endl;
        return make_pair(MatrixXd(), false);
    }

    int nrows, ncols;
    fin >> nrows >> ncols;

    MatrixXd A(nrows, ncols);
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            fin >> A(i, j);
        }
    }
    fin.close();

    return make_pair(A, true);
}

void dataPrinting(const char* eigenvectors_file, const char* eigenvalues_file, const MatrixXd& eigenvectors, const VectorXd& eigenvalues) {
    ofstream fout_eigenvectors(eigenvectors_file);
    if (!fout_eigenvectors.is_open()) {
        cerr << "Error: could not open output file " << eigenvectors_file << endl;
        return;
    }

    fout_eigenvectors << eigenvectors.transpose() << "\n";

    fout_eigenvectors.close();

    ofstream fout_eigenvalues(eigenvalues_file);
    if (!fout_eigenvalues.is_open()) {
        cerr << "Error: could not open output file " << eigenvalues_file << endl;
        return;
    }

    fout_eigenvalues << eigenvalues << "\n";

    fout_eigenvalues.close();
}

// Double es 64 bits y float es 32 bits. No se si esto puede impactar en la precisión.
// Igual el estandar de C++ es usar double. 
pair<MatrixXd,VectorXd> powerIteationwDeflation(MatrixXd A, int maxIter, double tol) {
    int n = A.cols(); // numero de autovalores. Esto no se si tiene sentido.;  

    // Para almacenar los autovalores y autovecotes de toda la matriz. 
    MatrixXd eigenvectors(n, n); 
    VectorXd eigenvalues(n);
    

    for (int i = 0; i < n; i++) {
        VectorXd x = VectorXd::Random(n);

        for (int j = 0; j < maxIter; j++) {
            
            VectorXd y = A * x;
            // Si Ax=lambda*x, entonces x es autovector y lambda es autovalor.
            // si y=Ax y Ax=lambda*x, entonces y=lambda*x y lambda=x'*y.
            double lambda = x.dot(y);
            x = y / y.norm(); 
            if (j > 0 && abs(lambda - eigenvalues(i)) < tol) {
                break;
            }
            eigenvalues[i] = lambda;
        }
        eigenvectors.col(i) = x;
        A -= eigenvalues(i) * x * x.transpose();
    }
return make_pair(eigenvectors, eigenvalues);
}



int main(int argc, char** argv) {
    if (argc != 6) {
        cerr << "Usage: " << argv[0] << " input_file values_out vectors_out iter tol" << std::endl;
        return 1;
    }

    const char* input_file = argv[1];
    const char* eigenvalues_file = argv[2];
    const char* eigenvectors_file = argv[3];
    int maxIter = atoi(argv[4]);
    double tol = atof(argv[5]);

    pair<MatrixXd, bool> data = dataloader(input_file);
    if (!data.second) {
        return 1;
    }

    MatrixXd A = data.first;

    cout << "Matrix A:" << endl << A << endl;

    pair<MatrixXd, VectorXd> results = powerIteationwDeflation(A, maxIter, tol);

    MatrixXd eigenvectors = results.first.transpose();
    VectorXd eigenvalues = results.second;

    dataPrinting(eigenvectors_file, eigenvalues_file, eigenvectors, eigenvalues);


    return 0;

}