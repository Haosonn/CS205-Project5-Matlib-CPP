#include <iostream>
#include <tuple>
#include "Matrix.h"
using namespace std;

void conversionTest() {
    cout << "Conversion test" << endl;
    Matrix<int> intMat = Matrix<int>::randomMatrix(3, 3);
    cout << "intMat:" << endl;
    intMat.printMatrix();
    Matrix<double> doubleMat = Matrix<double>::randomMatrix(3, 3);
    cout << "doubleMat:" << endl;
    doubleMat.printMatrix();
    cout << "intMat + doubleMat:" << endl;
    (intMat + doubleMat).printMatrix();
    cout << "doubleMat + intMat:" << endl;
    (doubleMat + intMat).printMatrix();
}

void inverseTest() {
    cout << "Inverse test" << endl;
    Matrix<double> doubleMatrix = Matrix<double>::randomMatrix(5, 5);
    doubleMatrix.printMatrix();
    Matrix<double> subDoubleMatrix = doubleMatrix.subMatrix(1, 2, 1, 2);
    subDoubleMatrix.printMatrix();
    Matrix<double> doubleInvMatrix = subDoubleMatrix.inverse();
    subDoubleMatrix.printMatrix();
    (subDoubleMatrix * doubleInvMatrix).printMatrix();
}

void PLUtest() {
    cout << "PLU test" << endl;
    Matrix<double> doubleMatrix = Matrix<double>::randomMatrix(5, 5);
    doubleMatrix.printMatrix();
    Matrix<double> subDoubleMatrix = doubleMatrix.subMatrix(1, 2, 1, 2);
    subDoubleMatrix.printMatrix();
    tuple<Matrix<double>, Matrix<double>, Matrix<double>> PLU = subDoubleMatrix.PLUFactorization();
    get<0>(PLU).printMatrix();
    get<1>(PLU).printMatrix();
    get<2>(PLU).printMatrix();
    (get<0>(PLU) * get<1>(PLU) * get<2>(PLU)).printMatrix();
}

void ImgTest() {
    cout << "Image test" << endl;
    Img<double> img(3, 5, 5);
    img.setChannel(0, Matrix<double>::randomMatrix(5, 5));
    img.setChannel(1, Matrix<double>::randomMatrix(5, 5));
    img.setChannel(2, Matrix<double>::randomMatrix(5, 5));
    img.printImg();
}

void mulTest() {
    cout << "Mul test" << endl;
    Matrix<double> doubleMatrix = Matrix<double>::randomMatrix(5, 5);
    cout << "doubleMatrix:" << endl;
    doubleMatrix.printMatrix();
    cout << "doubleMatrix * doubleMatrix:" << endl;
    (doubleMatrix * doubleMatrix).printMatrix();
}

int main() {
    conversionTest();
    inverseTest();
    PLUtest();
    ImgTest();
    mulTest();
}
