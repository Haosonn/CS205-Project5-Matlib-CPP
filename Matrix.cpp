#define PREC 1e-6

#include "Matrix.h"
#include <iostream>
#include <malloc.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>

#ifdef __OPENMP__
#include <omp.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
namespace avx {
    template <typename T>
    struct AVXTraits {static size_t size;};
    template <> struct AVXTraits<float> {
        using type = __m256;
        static size_t size;
    };
    template <> struct AVXTraits<double> {
        using type = __m256d;
        static size_t size;
    };
    template <> struct AVXTraits<int> {
        using type = __m256i;
        static size_t size;
    };
    template <> struct AVXTraits<short> {
        using type = __m256i;
        static size_t size;
    };
    template <> struct AVXTraits<unsigned char> {
        using type = __m256i;
        static size_t size;
    };
    size_t AVXTraits<float>::size = 8;
    size_t AVXTraits<double>::size = 4;
    size_t AVXTraits<int>::size = 8;
    size_t AVXTraits<short>::size = 0;
    size_t AVXTraits<unsigned char>::size = 0;

    template<class T>
    using AVXType = typename AVXTraits<T>::type;

    inline __m256 mm256_loadu(const float* a) {
        return _mm256_loadu_ps(a);
    }
    inline __m256d mm256_loadu(const double* a) {
        return _mm256_loadu_pd(a);
    }
    inline __m256i mm256_loadu(const int* a) {
        return _mm256_loadu_si256((__m256i const*)a);
    }
    inline __m256i mm256_loadu(unsigned char* a) {
        return _mm256_setzero_si256();
    }
    inline __m256i mm256_loadu(short* a) {
        return _mm256_setzero_si256();
    }

    inline __m256 mm256_mul(const __m256& m1, const __m256& m2) {
        return _mm256_mul_ps(m1, m2);
    }
    inline __m256d mm256_mul(const __m256d& m1, const __m256d& m2) {
        return _mm256_mul_pd(m1, m2);
    }
    inline __m256i mm256_mul(const __m256i& m1, const __m256i& m2) {
        return _mm256_mullo_epi32(m1, m2);
    }
}
#endif

#ifdef __NEON__
#include <arm_neon.h>
#endif

template <typename T>
MemBlock<T>::MemBlock(size_t stride, size_t size) : stride(stride), size(size), data(nullptr) {
    if (stride < 0 || size < 0) {
        std::cerr << "Callstack: MemBlock<T>::MemBlock(size_t stride, size_t size)" << std::endl;
        throw std::string("Exception in MemBlock<T>::MemBlock(size_t stride, size_t size), negative size");
        return;
    }
    if (!stride && !size)
        return;
    this->data = new T[size];
//    std::cout << "Allocated " << this->data << std::endl;
    if (this->data == nullptr) {
        std::cerr << "Callstack: MemBlock<T>::MemBlock(size_t stride, size_t size)" << std::endl;
        throw std::string("Exception in MemBlock<T>::MemBlock(size_t stride, size_t size), memory allocation failed");
    }
}
template <typename T>
void MemBlock<T>::release() {
    if (this->data != nullptr) {
//        std::cout << "Releasing memory block " << this->data << std::endl;
        delete[] this->data;
        this->data = nullptr;
    }
}
template <typename T>
MemBlock<T>::~MemBlock() {}

template <typename T>
MemBlock<T>& MemBlock<T>::operator=(const MemBlock<T> &m) {
    this->data = m.data;
    this->size = m.size;
    this->stride = m.stride;
    return *this;
}

template <typename T>
MatPtr<T>::MatPtr(size_t rows, size_t cols)
try : memBlock(cols, rows * cols), refCount(((rows > 0) && (cols > 0))? (new int(1)) : nullptr) {}
catch (std::string &e)  {
    std::cerr << "Callstack: MatPtr<T>::MatPtr(size_t rows, size_t cols)" << std::endl;
    throw e;
}

template <typename T>
MatPtr<T>::MatPtr(const MatPtr<T> &matPtr) {
    this->memBlock = matPtr.memBlock;
    this->refCount = matPtr.refCount;
    if(this->refCount != nullptr)
        (*this->refCount)++;
}

template <typename T>
MatPtr<T>::~MatPtr() {
    this->release();
}


template <typename T>
MatPtr<T>& MatPtr<T>::operator=(const MatPtr<T> &matPtr) {
    this->release();
    this->memBlock = matPtr.memBlock;
    this->refCount = matPtr.refCount;
    if(this->refCount != nullptr)
        (*this->refCount)++;
    return *this;
}

template <typename T>
void MatPtr<T>::release() {
    if (this->refCount != nullptr) {
        (*this->refCount)--;
        if (*this->refCount == 0) {
            delete this->refCount;
            this->memBlock.release();
        }
        this->refCount = nullptr;
    }
}


template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols)
try : rows(rows), cols(cols), bias(0), matPtr(rows, cols) {
    if (rows < 0 || cols < 0) {
        std::cerr << "Matrix size must be positive" << std::endl;
        rows = cols = 0;
        matPtr = MatPtr<T>(0, 0);
    }
//    std::cout << "Matrix<T>::Matrix(size_t rows, size_t cols)" << std::endl;
}
catch (std::string &e) {
    std::cerr << "Callstack: Matrix<T>::Matrix(size_t rows, size_t cols)" << std::endl;
    throw e;
}

template <typename T>
Matrix<T>::Matrix(const Matrix<T>& m) : rows(m.rows), cols(m.cols), bias(m.bias), matPtr(m.matPtr) {}

template <typename T>
Matrix<T> Matrix<T>::copy() const {
    Matrix<T> res(this->rows, this->cols);
    try {
        T * resData = res.matPtr.memBlock.data;
        T * data = this->matPtr.memBlock.data + this->bias;
        size_t stride = this->matPtr.memBlock.stride;
        for (size_t i = 0; i < this->rows; i++) {
            for (size_t j = 0; j < this->cols; j++) {
                resData[i * this->cols + j] = data[i * stride + j];
            }
        }
    }
    catch (std::string &e) {
        std::cerr << "Callstack: Matrix<T>::copy()" << std::endl;
        throw e;
    }
    return res;
}

template <typename T>
Matrix<T>::~Matrix() {
    this->matPtr.release();
}

template <typename T>
void Matrix<T>::printMatrix() const{
    if (this->matPtr.getData() == nullptr || this->rows == 0 || this->cols == 0) {
        std::cerr << "Callstack: Matrix<T>::printMatrix()" << std::endl;
        throw std::string("Exception in Matrix<T>::printMatrix(), matrix is empty");
    }
    T* data = this->matPtr.memBlock.data + this->bias;
    size_t stride = this->matPtr.memBlock.stride;
    for (size_t i = 0; i < this->rows; i++) {
        for (size_t j = 0; j < this->cols; j++) {
            std::cout << data[i * stride + j] << " ";
        }
        std::cout << std::endl;
    }
}

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &m) const {
    if (this->matPtr.getData() == nullptr || m.matPtr.getData() == nullptr) {
        std::cerr << "Callstack: Matrix<T>::operator+(const Matrix<T> &m)" << std::endl;
        throw std::string("Exception in Matrix<T>::operator+(const Matrix<T> &m), matrix is empty");
    }
    if (this->rows != m.rows || this->cols != m.cols) {
        std::cerr << "Callstack: Matrix<T>::operator+(const Matrix<T> &m)" << std::endl;
        throw std::string("Exception in Matrix<T>::operator+(const Matrix<T> &m), matrix sizes are not equal");
    }
    Matrix<T> result(this->rows, this->cols);
    size_t stride1 = this->matPtr.memBlock.stride;
    size_t stride2 = m.matPtr.memBlock.stride;
    T* data1 = this->matPtr.getData() + this->bias;
    T* data2 = m.matPtr.getData() + m.bias;
    T* data3 = result.matPtr.getData();
    for (size_t i = 0; i < this->rows; i++)
        for (size_t j = 0; j < this->cols; j++)
            data3[i * this->cols + j] = data1[i * stride1 + j] + data2[i * stride2 + j];
    return result;
}
template <typename T>
Matrix<T> Matrix<T>::operator+(const T &t) const {
    if (this->matPtr.getData() == nullptr) {
        std::cerr << "Callstack: Matrix<T>::operator+(const T &t)" << std::endl;
        throw std::string("Exception in Matrix<T>::operator+(const T &t), matrix is empty");
    }
    Matrix<T> result(this->rows, this->cols);
    size_t stride1 = this->matPtr.memBlock.stride;
    T* data1 = this->matPtr.getData() + this->bias;
    T* data3 = result.matPtr.getData();
    for (size_t i = 0; i < this->rows; i++)
        for (size_t j = 0; j < this->cols; j++)
            data3[i * this->cols + j] = data1[i * stride1 + j] + t;
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T> &m) const {
    if (this->matPtr.getData() == nullptr || m.matPtr.getData() == nullptr) {
        std::cerr << "Callstack: Matrix<T>::operator-(const Matrix<T> &m)" << std::endl;
        throw std::string("Exception in Matrix<T>::operator-(const Matrix<T> &m), matrix is empty");
    }
    if (this->rows != m.rows || this->cols != m.cols) {
        std::cerr << "Callstack: Matrix<T>::operator-(const Matrix<T> &m)" << std::endl;
        throw std::string("Exception in Matrix<T>::operator-(const Matrix<T> &m), matrix sizes are not equal");
    }
    Matrix<T> result(this->rows, this->cols);
    size_t stride1 = this->matPtr.memBlock.stride;
    size_t stride2 = m.matPtr.memBlock.stride;
    T* data1 = this->matPtr.getData() + this->bias;
    T* data2 = m.matPtr.getData() + m.bias;
    T* data3 = result.matPtr.getData();
    for (size_t i = 0; i < this->rows; i++)
        for (size_t j = 0; j < this->cols; j++)
            data3[i * this->cols + j] = data1[i * stride1 + j] - data2[i * stride2 + j];
    return result;
}
template <typename T>
Matrix<T> Matrix<T>::operator-(const T &t) const {
    if (this->matPtr.getData() == nullptr) {
        std::cerr << "Callstack: Matrix<T>::operator-(const T &t)" << std::endl;
        throw std::string("Exception in Matrix<T>::operator-(const T &t), matrix is empty");
    }
    Matrix<T> result(this->rows, this->cols);
    size_t stride1 = this->matPtr.memBlock.stride;
    T* data1 = this->matPtr.getData() + this->bias;
    T* data3 = result.matPtr.getData();
    for (size_t i = 0; i < this->rows; i++)
        for (size_t j = 0; j < this->cols; j++)
            data3[i * this->cols + j] = data1[i * stride1 + j] - t;
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &m) const {
    if (this->matPtr.getData() == nullptr || m.matPtr.getData() == nullptr) {
        std::cerr << "Callstack: Matrix<T>::operator*(const Matrix<T> &m)" << std::endl;
        throw std::string("Exception in Matrix<T>::operator*(const Matrix<T> &m), matrix is empty");
    }
    if (this->cols != m.rows) {
        std::cerr << "Callstack: Matrix<T>::operator*(const Matrix<T> &m)" << std::endl;
        throw std::string("Exception in Matrix<T>::operator*(const Matrix<T> &m), matrix sizes are not equal");
    }
    Matrix<T> result(this->rows, m.cols);
    size_t stride1 = this->matPtr.memBlock.stride;
    size_t stride2 = m.matPtr.memBlock.stride;
    T* data1 = this->matPtr.getData() + this->bias;
    T* data2 = m.matPtr.getData() + m.bias;
    T* data3 = result.matPtr.getData();
#ifdef __AVX2__
    size_t MAXSIZE = 512;
    memset(data3, 0, this->rows * m.cols * sizeof(T));
    if (this->cols < MAXSIZE && m.cols < MAXSIZE) {
        for (size_t i = 0; i < this->rows; i++) {
            for (size_t k = 0; k < this->cols; k++) {
                for (size_t j = 0; j < m.cols; j++) {
                    data3[i * m.cols + j] += data1[i * stride1 + k] * data2[k * stride2 + j];
                }
            }
        }
        return result;
    }
    size_t blockWidthLength = m.cols;
    size_t blockHeightLength = this->rows;
    if (blockWidthLength > MAXSIZE)
        blockWidthLength = MAXSIZE;
    if (blockHeightLength > MAXSIZE)
        blockHeightLength = MAXSIZE;
    size_t blockSize = blockWidthLength * blockHeightLength;
    T * ARearrange = (T *)malloc(this->rows * this->cols * sizeof(T));
    T * BRearrange = (T *)malloc(m.rows * m.cols * sizeof(T));
    for (size_t i = 0; i < this->rows; i++)
        memcpy(ARearrange + i * this->cols, data1 + i * stride1, this->cols * sizeof(T));
//        for (size_t j = 0; j < this->cols; j++)
//            ARearrange[i * this->cols + j] = data1[i * stride1 + j];
    for (size_t i = 0; i < m.rows; i++)
        for (size_t j = 0; j < m.cols; j++)
            BRearrange[j * m.rows + i] = data2[i * stride2 + j];
    size_t k = 0;
    T * ABlock = (T *)malloc(blockSize * sizeof(T));
    T * BBlock = (T *)malloc(blockSize * sizeof(T));
    T * CBlock = (T *)malloc(blockSize * sizeof(T));
    avx::AVXType<T> vecA;
    avx::AVXType<T> vecB;
    avx::AVXType<T> vecC;
    for (k = 0; k < this->cols; k += blockWidthLength) {
        size_t ag = 0;
        size_t bg = 0;
        for (ag = 0; ag <= this->rows - blockHeightLength; ag += blockHeightLength) {
            for (bg = 0; bg <= m.cols - blockHeightLength; bg += blockHeightLength) {
                size_t i = 0;
                size_t j = 0;
                memset(CBlock, 0, blockSize * sizeof(T));
                for (i = 0; i < blockHeightLength; i++) {
                    for (j = 0; j < blockWidthLength; j++) {
                        ABlock[i * blockWidthLength + j] = ARearrange[(ag + i) * this->cols + k + j];
                        BBlock[i * blockWidthLength + j] = BRearrange[(bg + i) * m.rows + k + j];
                    }
                }
                size_t l = 0;
                for (l = 0; l < blockWidthLength; l += avx::AVXTraits<T>::size) {
                    for (i = 0; i < blockHeightLength; i++) {
                        ABlock[i * blockWidthLength + l];
                        vecA = avx::mm256_loadu(ABlock + i * blockWidthLength + l);
                        for (j = 0; j < blockHeightLength; j++) {
                            vecB = avx::mm256_loadu(BBlock + j * blockWidthLength + l);
                            vecC = avx::mm256_mul(vecA, vecB);
                            T sum = vecC[0] + vecC[1] + vecC[2] + vecC[3];
                            if (avx::AVXTraits<T>::size == 8)
                                sum += vecC[4] + vecC[5] + vecC[6] + vecC[7];
                            CBlock[i * blockWidthLength + j] += sum;
                        }
                    }
                }
                for (;l < blockWidthLength; l++)
                    for (i = 0; i < blockHeightLength; i++)
                        for (j = 0; j < blockHeightLength; j++)
                            CBlock[i * blockWidthLength + j] += ABlock[i * blockWidthLength + l] * BBlock[j * blockWidthLength + l];
                for (i = 0; i < blockHeightLength; i++)
                    for (j = 0; j < blockHeightLength; j++)
                        data3[(ag + i) * m.cols + bg + j] += CBlock[i * blockWidthLength + j];
            }
        }
    }
    for (; k < this->cols; k++) {
        for (size_t i = 0; i < this->rows; i++) {
            for (size_t j = 0; j < m.cols; j++) {
                data3[i * m.cols + j] += ABlock[i * blockWidthLength + k] * BBlock[j * blockWidthLength + k];
            }
        }
    }
#else
    for (size_t i = 0; i < this->rows; i++) {
        for (size_t k = 0; k < this->cols; k++) {
            for (size_t j = 0; j < m.cols; j++) {
                data3[i * m.cols + j] += data1[i * stride1 + k] * data2[k * stride2 + j];
            }
        }
    }
#endif
    return result;
}
template <typename T>
Matrix<T> Matrix<T>::operator*(const T &t) const{
    if (this->matPtr.getData() == nullptr) {
        std::cerr << "Callstack: Matrix<T>::operator*(const T &t)" << std::endl;
        throw std::string("Exception in Matrix<T>::operator*(const T &t), matrix is empty");
    }
    Matrix<T> result(this->rows, this->cols);
    size_t stride1 = this->matPtr.memBlock.stride;
    T* data1 = this->matPtr.getData() + this->bias;
    T* data3 = result.matPtr.getData();
    for (size_t i = 0; i < this->rows; i++)
        for (size_t j = 0; j < this->cols; j++)
            data3[i * this->cols + j] = data1[i * stride1 + j] * t;
    return result;
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T> &m) {
    if(this != &m) {
        this->rows = m.rows;
        this->cols = m.cols;
        this->bias = m.bias;
        this->matPtr = m.matPtr;
    }
    return *this;
}

template <typename T>
bool Matrix<T>::operator==(const Matrix<T> &m) const {
    if (this->matPtr.getData() == nullptr || m.matPtr.getData() == nullptr) {
        std::cerr << "Callstack: Matrix<T>::operator*(const Matrix<T> &m)" << std::endl;
        throw std::string("Exception in Matrix<T>::operator*(const Matrix<T> &m), matrix is empty");
    }
    if (this->rows != m.rows || this->cols != m.cols)
        return false;
    if (this->matPtr.memBlock.data + this->bias == m.matPtr.memBlock.data + m.bias)
        return true;
    T * data1 = this->matPtr.getData() + this->bias;
    T * data2 = m.matPtr.getData() + m.bias;
    size_t stride1 = this->matPtr.memBlock.stride;
    size_t stride2 = m.matPtr.memBlock.stride;
    for (size_t i = 0; i < this->rows; i++)
        for (size_t j = 0; j < this->cols; j++)
            if (data1[i * stride1 + j] != data2[i * stride2 + j])
                return false;
    return true;
}

template <typename T>
T Matrix<T>::get(size_t i, size_t j) const {
    if (this->matPtr.getData() == nullptr) {
        std::cerr << "Callstack: Matrix<T>::get(size_t i, size_t j)" << std::endl;
        throw std::string("Exception in Matrix<T>::get(size_t i, size_t j), matrix is empty");
    }
    if (i >= this->rows || j >= this->cols || i < 0 || j < 0) {
        std::cerr << "Callstack: Matrix<T>::get(size_t i, size_t j)" << std::endl;
        throw std::string("Exception in Matrix<T>::get(size_t i, size_t j), index out of range");
    }
    T * data = this->matPtr.getData() + this->bias;
    size_t stride = this->matPtr.memBlock.stride;
    return data[i * stride + j];
}

template <typename T>
Matrix<T> Matrix<T>::randomMatrix(size_t rows, size_t cols) {
    try {
        Matrix<T> m(rows, cols);
        T* dataPtr = m.matPtr.getData();
        size_t size = rows * cols;
        for (size_t i = 0; i < size; i++)
            dataPtr[i] = (rand() % 100) / 10.0;
        return m;
    } catch (std::string& e) {
        std::cerr << "Callstack: Matrix<T>::randomMatrix(size_t rows, size_t cols)" << std::endl;
        throw e;
    }
}

template <typename T>
Matrix<T> Matrix<T>::identityMatrix(size_t size) {
    try {
        Matrix<T> m(size, size);
        T *dataPtr = m.matPtr.getData();
        memset(dataPtr, 0, size * size * sizeof(T));
        for (size_t i = 0; i < size; i++)
            dataPtr[i * size + i] = 1;
        return m;
    } catch (std::string &e) {
        std::cerr << "Callstack: Matrix<T>::identityMatrix(size_t size)" << std::endl;
        throw e;
    }
}

template<typename T>
Matrix<T> Matrix<T>::subMatrix(size_t beginRow, size_t endRow, size_t beginCol, size_t endCol) {
    if (beginRow > endRow || beginCol > endCol || endRow > this->rows || endCol > this->cols) {
        std::cerr << "Callstack: Matrix<T>::subMatrix(size_t beginRow, size_t endRow, size_t beginCol, size_t endCol)" << std::endl;
        throw std::string("Exception in Matrix<T>::subMatrix(size_t beginRow, size_t endRow, size_t beginCol, size_t endCol): Invalid range");
    }
    Matrix<T> res(*this);
    res.rows = endRow - beginRow + 1;
    res.cols = endCol - beginCol + 1;
    res.bias += beginRow * this->matPtr.memBlock.stride + beginCol;
    return res;
}

template <typename T>
void Matrix<T>::rowSwap(size_t row1, size_t row2) {
    if (this->matPtr.getData() == nullptr) {
        std::cerr << "Callstack: Matrix<T>::rowSwap(size_t row1, size_t row2)" << std::endl;
        throw std::string("Exception in Matrix<T>::rowSwap(size_t row1, size_t row2): Matrix is empty");
    }
    if (row1 >= this->rows || row2 >= this->rows || row1 < 0 || row2 < 0) {
        std::cerr << "Callstack: Matrix<T>::rowSwap(size_t row1, size_t row2)" << std::endl;
        throw std::string("Exception in Matrix<T>::rowSwap(size_t row1, size_t row2): Invalid row index");
    }
    T * data = this->matPtr.getData() + this->bias;
    size_t stride = this->matPtr.memBlock.stride;
    for (size_t i = 0; i < this->cols; i++) {
        T temp = data[row1 * stride+ i];
        data[row1 * stride + i] = data[row2 * stride + i];
        data[row2 * stride + i] = temp;
    }
}

template <typename T>
void Matrix<T>::rowCombination(size_t row1, size_t row2, T scalar1, T scalar2, size_t dstRow) {
    if (this->matPtr.getData() == nullptr) {
        std::cerr << "Callstack: Matrix<T>::rowCombination(size_t row1, size_t row2, T scalar1, T scalar2, size_t dstRow)" << std::endl;
        throw std::string("Exception in Matrix<T>::rowCombination(size_t row1, size_t row2, T scalar1, T scalar2, size_t dstRow): Matrix is empty");
    }
    if (row1 >= this->rows || row2 >= this->rows || dstRow >= this->rows || row1 < 0 || row2 < 0 || dstRow < 0) {
        std::cerr << "Callstack: Matrix<T>::rowCombination(size_t row1, size_t row2, T scalar1, T scalar2, size_t dstRow)" << std::endl;
        throw std::string("Exception in Matrix<T>::rowCombination(size_t row1, size_t row2, T scalar1, T scalar2, size_t dstRow): Invalid row index");
    }
    T * data = this->matPtr.getData() + this->bias;
    size_t stride = this->matPtr.memBlock.stride;
    for (size_t i = 0; i < this->cols; i++) {
        data[dstRow * stride + i] = scalar1 * data[row1 * stride + i] + scalar2 * data[row2 * stride + i];
    }
}

template <typename T>
void Matrix<T>::rowMultiplication(size_t row, T scalar) {
    if (this->matPtr.getData() == nullptr) {
        std::cerr << "Callstack: Matrix<T>::rowMultiplication(size_t row, T scalar)" << std::endl;
        throw std::string("Exception in Matrix<T>::rowMultiplication(size_t row, T scalar): Matrix is empty");
    }
    if (row >= this->rows || row < 0) {
        std::cerr << "Callstack: Matrix<T>::rowMultiplication(size_t row, T scalar)" << std::endl;
        throw std::string("Exception in Matrix<T>::rowMultiplication(size_t row, T scalar): Invalid row index");
    }
    T * data = this->matPtr.getData() + this->bias;
    size_t stride = this->matPtr.memBlock.stride;
    for (size_t i = 0; i < this->cols; i++)
        data[row * stride + i] *= scalar;
}

template<typename T>
double Matrix<T>::determinant() const {
    if (this->matPtr.getData() == nullptr) {
        std::cerr << "Callstack: Matrix<T>::determinant()" << std::endl;
        throw std::string("Exception in Matrix<T>::determinant(): Matrix is empty");
    }
    if (this->rows != this->cols) {
        std::cerr << "Callstack: Matrix<T>::determinant()" << std::endl;
        throw std::string("Exception in Matrix<T>::determinant(): Matrix is not square");
    }
    if (this->rows == 1) {
        return (this->matPtr.getData() + this->bias)[0];
    }
    double sum = 0;
    Matrix sub(this->rows - 1, this->cols - 1);
    T * subData = sub.matPtr.getData();
    T * data = this->matPtr.getData();
    size_t stride = this->matPtr.memBlock.stride;
    for (size_t i = 0; i < this->cols; i++) {
        for (size_t j = 1; j < this->rows; j++) {
            for (size_t k = 0; k < this->cols; k++) {
                if (k < i) {
                    subData[(j - 1) * sub.cols + k] = data[j * stride + k];
                } else if (k > i) {
                    subData[(j - 1) * sub.cols + k - 1] = data[j * stride + k];
                }
            }
        }
        double det = sub.determinant();
        sum = sum + (T) (i & 1 ? -1 : 1) * data[i] * det;
    }
    return sum;
}

template<typename T>
Matrix<T> Matrix<T>::rowEchelon() const {
    if (this->matPtr.getData() == nullptr) {
        std::cerr << "Callstack: Matrix<T>::rowEchelon()" << std::endl;
        throw std::string("Exception in Matrix<T>::rowEchelon(): Matrix is empty");
    }

    Matrix<T> res = this->copy();
    res.printMatrix();
    for (size_t i = res.cols + 1; i < res.rows; i++) {
        res.rowMultiplication(i, 0);
    }
    T * data = res.matPtr.getData();
    for (size_t i = 0; i < res.rows; i++) {
        if(fabsf(data[i * res.cols + i]) < PREC) {
            for (size_t j = i + 1; j < res.rows; j++){
                if(fabsf(data[j * res.cols + i]) > PREC) {
                    res.rowSwap(i, j);
                    break;
                }
            }
        }
        if(fabsf(data[i * res.cols + i]) < PREC)
            continue;
        else {
            for (size_t j = i + 1; j < res.rows; j++) {
                if(fabsf(data[i * res.cols + j]) == 0) {
                    std::cerr << "Callstack: Matrix<T>::rowEchelon()" << std::endl;
                    throw std::string("Exception in Matrix<T>::rowEchelon(): Divided by zero");
                }
                T coef = data[j * res.cols + i] / data[i * res.cols + i];
                res.rowCombination(i, j, coef, -1, j);
            }
        }
    }
    return res;
}

template<typename T>
Matrix<T> Matrix<T>::rowReducedEchelon() const {
    if (this->matPtr.getData() == nullptr) {
        std::cerr << "Callstack: Matrix<T>::rowReducedEchelon()" << std::endl;
        throw std::string("Exception in Matrix<T>::rowReducedEchelon(): Matrix is empty");
    }
    try {
        this->printMatrix();
        Matrix<T> res = this->rowEchelon();
        size_t rank = 0;
        size_t squareLength = (res.rows < res.cols) ? res.rows : res.cols;
        T * data = res.matPtr.getData();
        for (size_t i = 0; i < squareLength; i++)
            rank += (fabsf(data[i * res.cols + i]) > PREC);
        for (size_t i = 0; i < rank; i++) {
            if ((data[i * res.cols + i] == 0)) {
                throw std::string("Exception in Matrix<T>::rowReducedEchelon(): Divided by zero");
            }
            res.rowMultiplication(i, 1 / data[i * res.cols + i]);
        }
        for (size_t i = 0; i < rank; i++)
            for (size_t j = i + 1; j < rank; j++)
                res.rowCombination(i, j, 1, -data[i * res.cols + j], i);
        return res;
    } catch (std::string & e) {
        std::cerr << "Callstack: Matrix<T>::rowReducedEchelon()" << std::endl;
        throw e;
    }

}

template<typename T>
size_t Matrix<T>::rank() const {
    try {
        Matrix obj = this->rowEchelon();
        size_t rank = 0;
        T * data = obj.matPtr.getData();
        for (size_t i = 1; i <= (obj.rows > obj.cols ? obj.cols : obj.rows); i++)
            rank += (fabsf(data[i * obj.cols + i]) > PREC);
        return rank;
    }catch (std::string & e) {
        std::cerr << "Callstack: Matrix<T>::rank()" << std::endl;
        throw e;
    }
}

template<typename T>
Matrix<T> Matrix<T>::transpose() const {
    if (this->matPtr.getData() == nullptr) {
        std::cerr << "Callstack: Matrix<T>::transpose()" << std::endl;
        throw std::string("Exception in Matrix<T>::transpose(): Matrix is empty");
    }
    try {
        Matrix<T> res(this->cols, this->rows);
        T * data = this->matPtr.getData() + this->bias;
        size_t stride = this->matPtr.memBlock.stride;
        T * resData = res.matPtr.getData() + res.bias;
        for (size_t i = 0; i < this->rows; i++)
            for (size_t j = 0; j < this->cols; j++)
                resData[j * res.cols + i] = data[i * stride + j];
        return res;
    } catch (std::string & e) {
        std::cerr << "Callstack: Matrix<T>::transpose()" << std::endl;
        throw e;
    }
}

template<typename T>
Matrix<T> Matrix<T>::inverse() const {
    if (this->matPtr.getData() == nullptr) {
        std::cerr << "Callstack: Matrix<T>::inverse()" << std::endl;
        throw std::string("Exception in Matrix<T>::inverse(): Matrix is empty");
    }
    if (this->rows != this->cols) {
        std::cerr << "Callstack: Matrix<T>::inverse()" << std::endl;
        throw std::string("Exception in Matrix<T>::inverse(): Matrix is not square");
    }
    double det;
    try {
        det = this->determinant();
    } catch (std::string &e) {
        std::cerr << "Callstack: Matrix<T>::inverse()" << std::endl;
        throw e;
    }
    if (fabsf(det) < PREC) {
        std::cerr << "Callstack: Matrix<T>::inverse()" << std::endl;
        throw std::string("Exception in Matrix<T>::inverse(): Matrix is singular");
    }
    Matrix aug(this->rows, this->cols * 2);
    Matrix rref(this->rows, this->cols * 2);
    T *augData = aug.matPtr.getData();
    T *data = this->matPtr.getData() + this->bias;
    size_t stride = this->matPtr.memBlock.stride;
    for (size_t i = 0; i < this->rows; i++)
        for (size_t j = 0; j < this->cols; j++)
            augData[i * aug.cols + j] = data[i * stride + j];
    for (size_t i = 0; i < this->rows; i++)
        for (size_t j = this->cols; j < 2 * this->cols; j++)
            augData[i * aug.cols + j] = (i == j - this->cols) ? 1 : 0;
    try {
        rref = aug.rowReducedEchelon();
    } catch (std::string &e) {
        std::cerr << "Callstack: Matrix<T>::inverse()" << std::endl;
        throw e;
    }
    try {
        Matrix<T> res(this->rows, this->cols);
        res = rref.subMatrix(0, this->rows - 1, this->cols, 2 * this->cols - 1);
        return res;
    } catch (std::string &e) {
        std::cerr << "Callstack: Matrix<T>::inverse()" << std::endl;
        throw e;
    }
}

template<typename T>
std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> Matrix<T>::PLUFactorization() const {
    if (this->matPtr.getData() == nullptr) {
        std::cerr << "Callstack: Matrix<T>::PLUFactorization()" << std::endl;
        throw std::string("Exception in Matrix<T>::PLUFactorization(): Matrix is empty");
    }
    Matrix<T> P = identityMatrix(this->rows);
    Matrix<T> L = identityMatrix(this->rows);
    Matrix<T> U = this->copy();
    T * UData = U.matPtr.getData();
    T * LData = L.matPtr.getData();
    for (size_t i = 0; i < U.rows; i++) {
        if (i >= U.cols) {
            U.rowMultiplication(i, 0);
            continue;
        }
        if(fabsf(UData[i * U.cols + i]) < PREC) {
            for (size_t j = i + 1; j < U.rows; j++){
                if(fabsf(UData[j * U.cols + i]) >= PREC) {
                    U.rowSwap(i, j);
                    P.rowSwap(i, j);
                    break;
                }
            }
        }
        if(fabsf(UData[i * U.cols + i]) < PREC)
            continue;
        else {
            for (size_t j = i + 1; j < U.rows; j++) {
                float coef = UData[j * U.cols + i] / UData[i * U.cols + i];
                U.rowCombination(i, j, -coef, 1, j);
                LData[j * L.cols + i] = coef;
            }
        }
    }
    return std::make_tuple(P, L, U);
}



/*
template<typename T>
std::tuple<Matrix<T>, Matrix<T>> Matrix<T>::QRFactorization() const {
    if (this->matPtr.getData() == nullptr) {
        std::cerr << "Callstack: Matrix<T>::QRFactorization()" << std::endl;
        throw std::string("Exception in Matrix<T>::QRFactorization(): Matrix is empty");
    }
    if (this->rows < this->cols) {
        std::cerr << "Callstack: Matrix<T>::QRFactorization()" << std::endl;
        throw std::string("Exception in Matrix<T>::QRFactorization(): Matrix is not tall");
    }
    Matrix<T> Q(this->rows, this->cols);
    Matrix<T> R(this->cols, this->cols);
    T * v = (T *)malloc(sizeof(T) * (R.rows));
    T * u = (T *)malloc(sizeof(T) * (R.rows));
    T * data = this->matPtr.getData() + this->bias;
    T * RData = R.matPtr.getData();
    size_t stride = this->matPtr.memBlock.stride;
    for (int i = 0; i < this->cols; i++) {
        for (int j = 0; j < this->rows; j++) {
            v[j] = data[j * stride + i];
        }
        for (int j = 1; j < i; j++) {
            for (int k = 1; k <= R->row; k++) {
                u[k - 1] = Q->data[idx(k, j, Q)];
            }
            float dotProductVU = dotProduct(v, u, R->row);
            float dotProductUU = dotProduct(u, u, R->row);
            if (dotProductUU == 0) {
                deleteMatrix(Q);
                deleteMatrix(R);
                free(v);
                free(u);
                return MAT_DIVIDE_BY_ZERO;
            }
            float coef = dotProductVU / dotProductUU;
            for (int k = 1; k <= R->row; k++) {
                v[k - 1] -= coef * u[k - 1];
            }
        }
        float norm = sqrtf(dotProduct(v, v, R->row));
        for (int j = 1; j <= R->row; j++) {
            Q->data[idx(j, i, Q)] = v[j - 1] / norm;
        }
        for (int j = 1; j <= R->row; j++) {
            v[j - 1] = R->data[idx(j, i, R)];
        }
        //R_ij = q_i^T * a_j
        for (int k = 1; k <= R->row; k++) {
            u[k - 1] = mat->data[idx(k, i, Q)];
        }
        for (int j = 1; j < i; j++) {
            for (int k = 1; k <= R->row; k++) {
                v[k - 1] = Q->data[idx(k, j, mat)];
            }
            R->data[idx(j, i, R)] = dotProduct(v, u, R->row);
        }
        R->data[idx(i, i, R)] = norm;
    }
}
 */

template class MemBlock<int>;
template class MatPtr<int>;
template class Matrix<int>;
template class Img<int>;
template class MemBlock<short>;
template class MatPtr<short>;
template class Matrix<short>;
template class Img<short>;
template class MemBlock<float>;
template class MatPtr<float>;
template class Matrix<float>;
template class Img<float>;
template class MemBlock<double>;
template class MatPtr<double>;
template class Matrix<double>;
template class Img<double>;
template class MemBlock<unsigned char>;
template class MatPtr<unsigned char>;
template class Matrix<unsigned char>;
template class Img<unsigned char>;