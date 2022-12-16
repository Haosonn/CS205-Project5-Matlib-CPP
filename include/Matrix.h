#ifndef MATRIXLIB_MATRIX_H
#define MATRIXLIB_MATRIX_H

#include <iostream>
#include <tuple>


template <typename T>
class MemBlock {
public:
    T* data;
    size_t stride;
    size_t size;
    MemBlock(size_t stride = 0, size_t size = 0);
    ~MemBlock();
    MemBlock<T>& operator= (const MemBlock<T>& m);
    void release();
};

template <typename T>
class MatPtr {
public:
    MemBlock<T> memBlock;
    int * refCount;
    MatPtr(size_t rows, size_t cols);
    MatPtr(const MatPtr& matPtr);
    ~MatPtr();
    T* getData() const{
        return this->memBlock.data;
    }
    void release();
    MatPtr& operator=(const MatPtr& matPtr);
};


template <typename T>
class Matrix {
public:
    size_t rows;
    size_t cols;
    size_t bias;
    MatPtr<T> matPtr;
    Matrix(): rows(0), cols(0), bias(0), matPtr(0, 0) {};
    Matrix(size_t rows, size_t cols);
    Matrix(const Matrix<T>& m);

    Matrix copy() const;

    template<class U>
    Matrix(const Matrix<U>& m) : rows(m.rows), cols(m.cols), bias(0), matPtr(m.rows, m.cols) {
        T * dataFrom = this->matPtr.getData();
        U * dataTo = m.matPtr.getData();
        for(size_t i = 0; i < this->rows; i++){
            for(size_t j = 0; j < this->cols; j++){
                dataFrom[i * this->cols + j] = (T)dataTo[i * m.matPtr.memBlock.stride + j];
            }
        }
    }
    ~Matrix();
    void printMatrix() const;
    Matrix<T>& operator=(const Matrix<T>& m);
    Matrix<T> operator+(const Matrix<T>& m) const;
    Matrix<T> operator+(const T& t) const;
    Matrix<T> operator-(const Matrix<T>& m) const;
    Matrix<T> operator-(const T& t) const;
    Matrix<T> operator*(const Matrix<T>& m) const;
    Matrix<T> operator*(const T& t) const;
    bool operator==(const Matrix<T>& m) const;

    /*
     * @brief return the value in corresponding position
     * @param i, j: the position of the value
     * @note: the position is counted from 0
     */
    T get(size_t i, size_t j) const;

    /*
     * @brief return the inverse of the matrix
     */
    Matrix<T> inverse() const;
    /*
     * @brief: return the transpose of the matrix
     */
    Matrix<T> transpose() const;

    /*
     * @brief: modify a row of the matrix by row combination
     * @param: dstRow: the row to be modified
     * @param: row1, row2, scalar1, scalar2: row1 * scalar1 + row2 * scalar2 = dstRow
     */
    void rowCombination(size_t row1, size_t row2, T scalar1, T scalar2, size_t dstRow);
    /*
     * @brief: modify a row of the matrix by multiplication
     * @param: row: the row to be modified
     * @param: scalar: row * scalar = row
     */
    void rowMultiplication(size_t row, T scalar);
    /*
     * @brief: modify a row of the matrix by swap
     * @param: row1, row2: row1 <-> row2
     */
    void rowSwap(size_t row1, size_t row2);

    /*
     * @brief: return row echelon form of the matrix
     */
    Matrix<T> rowEchelon() const;
    /*
     * @brief: return reduced row echelon form of the matrix
     */
    Matrix<T> rowReducedEchelon() const;

    /*
     * @brief: return the PLU decomposition of the matrix (P, L, U) as a tuple
     */
    std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> PLUFactorization() const;

    /*
     * @brief: return the determinant of the matrix
     */
    double determinant() const;
    /*
     * @brief: return the rank of the matrix
     */
    size_t rank() const;

    /*
     * @brief: return the submatrix of the matrix
     * @param: rowStart, rowEnd, colStart, colEnd: the range of the submatrix
     * @note: The submatrix shares the same memory with the original matrix
     *        i.e., modifying the submatrix will also modify the original matrix
     */
    Matrix<T> subMatrix(size_t beginRow, size_t endRow, size_t beginCol, size_t endCol);
    /*
     * @brief: return a random matrix
     * @param: rows, cols: the size of the matrix
     */
    static Matrix<T> randomMatrix(size_t rows, size_t cols);
    /*
     * @brief: return a identity matrix
     * @param: size: the size of the matrix
     */
    static Matrix<T> identityMatrix(size_t size);
};


template <typename T>
class Img {
public:
    Matrix<T>* img;
    size_t channels;
    size_t rows;
    size_t cols;
    Img(): img(nullptr), channels(0), rows(0), cols(0) {};
    Img(size_t channels, size_t rows, size_t cols) : channels(channels), rows(rows), cols(cols) {
        img = new Matrix<T>[channels];
        for(size_t i = 0; i < channels; i++){
            img[i] = Matrix<T>(rows, cols);
        }
    }
    ~Img() {
        delete[] img;
        img = nullptr;
    }
    Img(const Img& img) : channels(img.channels), rows(img.rows), cols(img.cols) {
        this->img = new Matrix<T>[channels];
        for(size_t i = 0; i < channels; i++){
            this->img[i] = img.img[i];
        }
    }
    void setChannel(size_t channel, const Matrix<T>& m){
        if(channel >= this->channels){
            std::cerr << "Callstack: Img::setChannel(size_t channel, const Matrix<T>& m)" << std::endl;
            throw std::string("Error in Img::setChannel(size_t channel, const Matrix<T>& m): channel out of range");
        }
        if(m.rows != this->rows || m.cols != this->cols){
            std::cerr << "Callstack: Img::setChannel(size_t channel, const Matrix<T>& m)" << std::endl;
            throw std::string("Error in Img::setChannel(size_t channel, const Matrix<T>& m): matrix size mismatch");
        }
        this->img[channel] = m;
    }
    Matrix<T> getChannel(size_t channel) const{
        if(channel >= this->channels){
            std::cerr << "Callstack: Img::getChannel(size_t channel) const" << std::endl;
            throw std::string("Error in Img::getChannel(size_t channel) const: channel out of range");
        }
        return this->img[channel];
    }
    void printImg() const{
        T ** data = new T*[this->channels];
        for (size_t i = 0; i < this->channels; i++){
            data[i] = this->img[i].matPtr.getData();
            if (data[i] == nullptr) {
                std::cerr << "Callstack: Img::printImg() const" << std::endl;
                throw std::string("Error in Img::printImg() const: One of the channels is empty");
            }
            data[i] += this->img[i].bias;
        }
        for(size_t i = 0; i < this->rows; i++){
            for(size_t j = 0; j < this->cols; j++){
                std::cout << "(";
                for (size_t k = 0; k < this->channels; k++){
                    if (k > 0) std::cout << ", ";
                    std::cout << data[k][i * this->img[k].matPtr.memBlock.stride + j];
                }
                std::cout << ") ";
            }
            std::cout << std::endl;
        }
    }
};

#endif
