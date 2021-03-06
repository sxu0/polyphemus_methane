// Copyright (C) 2003-2009 Marc Duruflé
// Copyright (C) 2001-2009 Vivien Mallet
//
// This file is part of the linear-algebra library Seldon,
// http://seldon.sourceforge.net/.
//
// Seldon is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation; either version 2.1 of the License, or (at your option)
// any later version.
//
// Seldon is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
// more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with Seldon. If not, see http://www.gnu.org/licenses/.


#ifndef SELDON_FILE_MATRIX_ARRAY_COMPLEX_SPARSE_CXX

#include "Matrix_ArrayComplexSparse.hxx"

namespace Seldon
{


  /****************
   * CONSTRUCTORS *
   ****************/


  //! Default constructor.
  /*!
    Builds a empty matrix.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::
  Matrix_ArrayComplexSparse()
    : val_real_(), val_imag_()
  {
    this->m_ = 0;
    this->n_ = 0;
  }


  //! Constructor.
  /*!
    Builds an empty i by j sparse matrix.
    \param i number of rows.
    \param j number of columns.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::
  Matrix_ArrayComplexSparse(int i, int j):
    val_real_(Storage::GetFirst(i, j)), val_imag_(Storage::GetFirst(i, j))
  {
    this->m_ = i;
    this->n_ = j;
  }


  /**************
   * DESTRUCTOR *
   **************/


  //! Destructor.
  template <class T, class Prop, class Storage, class Allocator>
  inline Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::
  ~Matrix_ArrayComplexSparse()
  {
    this->m_ = 0;
    this->n_ = 0;
  }


  //! Clears the matrix.
  /*! This methods is equivalent to the destructor. On exit, the matrix is
    empty (0 by 0).
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline void Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::Clear()
  {
    this->~Matrix_ArrayComplexSparse();
  }


  /*********************
   * MEMORY MANAGEMENT *
   *********************/


  //! Reallocates memory to resize the matrix.
  /*!
    On exit, the matrix is a i x j matrix.
    \param i number of rows.
    \param j number of columns.
    \warning Data is lost.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline void Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::
  Reallocate(int i, int j)
  {
    // Clears previous entries.
    Clear();

    this->m_ = i;
    this->n_ = j;

    int n = Storage::GetFirst(i,j);
    val_real_.Reallocate(n);
    val_imag_.Reallocate(n);
  }


  //! Reallocates additional memory to resize the matrix.
  /*!
    On exit, the matrix is a i x j matrix.
    \param i number of rows.
    \param j number of columns.
    \note Data is kept.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline void Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::
  Resize(int i, int j)
  {
    int n = Storage::GetFirst(this->m_, n_);
    int new_n = Storage::GetFirst(i, j);
    if (n != new_n)
      {
	Vector<Vector<T, VectSparse, Allocator>, VectFull,
	  NewAlloc<Vector<T, VectSparse, Allocator> > > new_val_real;

	Vector<Vector<T, VectSparse, Allocator>, VectFull,
	  NewAlloc<Vector<T, VectSparse, Allocator> > > new_val_imag;

	new_val_real.Reallocate(new_n);
	new_val_imag.Reallocate(new_n);

	for (int k = 0 ; k < min(n, new_n) ; k++)
	  {
	    Swap(new_val_real(k), this->val_real_(k));
	    Swap(new_val_imag(k), this->val_imag_(k));
	  }

	val_real_.SetData(new_n, new_val_real.GetData());
	val_imag_.SetData(new_n, new_val_imag.GetData());
	new_val_real.Nullify();
	new_val_imag.Nullify();

      }

    this->m_ = i;
    this->n_ = j;
  }


  /*******************
   * BASIC FUNCTIONS *
   *******************/


  //! Returns the number of rows.
  /*!
    \return the number of rows.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline int Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>
  ::GetM() const
  {
    return m_;
  }


  //! Returns the number of columns.
  /*!
    \return the number of columns.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline int Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>
  ::GetN() const
  {
    return n_;
  }


  //! Returns the number of rows of the matrix possibly transposed.
  /*!
    \param status assumed status about the transposition of the matrix.
    \return The number of rows of the possibly-transposed matrix.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline int Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>
  ::GetM(const SeldonTranspose& status) const
  {
    if (status.NoTrans())
      return m_;
    else
      return n_;
  }


  //! Returns the number of columns of the matrix possibly transposed.
  /*!
    \param status assumed status about the transposition of the matrix.
    \return The number of columns of the possibly-transposed matrix.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline int Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>
  ::GetN(const SeldonTranspose& status) const
  {
    if (status.NoTrans())
      return n_;
    else
      return m_;
  }


  //! Returns the number of non-zero elements (real part).
  /*!
    \return The number of non-zero elements for real part of matrix.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline int Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::
  GetRealNonZeros() const
  {
    int nnz = 0;
    for (int i = 0; i < this->val_real_.GetM(); i++)
      nnz += this->val_real_(i).GetM();

    return nnz;
  }


  //! Returns the number of non-zero elements (imaginary part).
  /*!
    \return The number of non-zero elements for imaginary part of matrix.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline int Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::
  GetImagNonZeros() const
  {
    int nnz = 0;
    for (int i = 0; i < this->val_imag_.GetM(); i++)
      nnz += this->val_imag_(i).GetM();

    return nnz;
  }


  //! Returns the number of elements stored in memory (real part).
  /*!
    Returns the number of elements stored in memory, i.e.
    the number of non-zero entries.
    \return The number of elements stored in memory.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline int Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::
  GetRealDataSize() const
  {
    return GetRealNonZeros();
  }


  //! Returns the number of elements stored in memory (imaginary part).
  /*!
    Returns the number of elements stored in memory, i.e.
    the number of non-zero entries.
    \return The number of elements stored in memory.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline int Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::
  GetImagDataSize() const
  {
    return GetImagNonZeros();
  }


  //! Returns the number of elements stored in memory (real+imaginary part).
  /*!
    Returns the number of elements stored in memory, i.e.
    the number of non-zero entries.
    \return The number of elements stored in memory.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline int Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::
  GetDataSize() const
  {
    return (GetRealNonZeros()+GetImagNonZeros());
  }


  //! Returns column indices of non-zero entries in row (real part).
  /*!
    \param[in] i row number.
    \return The array of column indices of non-zero entries
    of row i.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline int* Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::
  GetRealInd(int i) const
  {
    return val_real_(i).GetIndex();
  }


  //! Returns values of non-zero entries of a row (real part).
  /*!
    \param[in] i row number.
    \return The array of values of non-zero entries of row i.
  */
  template <class T, class Prop, class Storage, class Allocator> inline T*
  Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::GetRealData(int i)
    const
  {
    return val_real_(i).GetData();
  }


  //! Returns column indices of non-zero entries in row (imaginary part).
  /*!
    \param[in] i row number.
    \return the array of column indices of non-zero entries
    of row i.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline int* Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::
  GetImagInd(int i) const
  {
    return val_imag_(i).GetIndex();
  }


  //! Returns values of non-zero entries of a row (imaginary part).
  /*!
    \param[in] i row number.
    \return The array of values of non-zero entries of row i.
  */
  template <class T, class Prop, class Storage, class Allocator> inline T*
  Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::GetImagData(int i)
    const
  {
    return val_imag_(i).GetData();
  }


  /**********************************
   * ELEMENT ACCESS AND AFFECTATION *
   **********************************/


  //! Access operator.
  /*!
    Returns the value of element (i, j).
    \param i row index.
    \param j column index.
    \return Element (i, j) of the matrix.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline complex<T>
  Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::operator()
    (int i, int j) const
  {

#ifdef SELDON_CHECK_BOUNDS
    if (i < 0 || i >= this->m_)
      throw WrongRow("Matrix::operator()", "Index should be in [0, "
		     + to_str(this->m_-1) + "], but is equal to "
		     + to_str(i) + ".");

    if (j < 0 || j >= this->n_)
      throw WrongCol("Matrix::operator()", "Index should be in [0, "
		     + to_str(this->n_-1) + "], but is equal to "
		     + to_str(j) + ".");
#endif

    return complex<T>(this->val_real_(Storage::GetFirst(i, j))
		      (Storage::GetSecond(i, j)),
		      this->val_imag_(Storage::GetFirst(i, j))
		      (Storage::GetSecond(i, j)) );
  }


  //! Returns j-th non-zero value of row i (real part).
  /*!
    \param[in] i row number.
    \param[in] j local number.
    \return j-th non-zero entry of row i.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline const T& Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::
  ValueReal(int i, int j) const
  {

#ifdef SELDON_CHECK_BOUNDS
    if (i < 0 || i >= this->m_)
      throw WrongRow("Matrix_ArraySparse::value", "Index should be in [0, "
		     + to_str(this->m_-1) + "], but is equal to "
		     + to_str(i) + ".");

    if (j < 0 || j >= this->val_real_(i).GetM())
      throw WrongCol("Matrix_ArraySparse::value", "Index should be in [0, " +
		     to_str(this->val_real_(i).GetM()-1) + "], but is equal to "
		     + to_str(j) + ".");
#endif

    return val_real_(i).Value(j);
  }


  //! Returns j-th non-zero value of row i (real part).
  /*!
    \param[in] i row number.
    \param[in] j local number.
    \return j-th non-zero entry of row i.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline T&
  Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::
  ValueReal(int i, int j)
  {

#ifdef SELDON_CHECK_BOUNDS
    if (i < 0 || i >= this->m_)
      throw WrongRow("Matrix_ArraySparse::value", "Index should be in [0, "
		     + to_str(this->m_-1) + "], but is equal to "
		     + to_str(i) + ".");
    if (j < 0 || j >= this->val_real_(i).GetM())
      throw WrongCol("Matrix_ArraySparse::value", "Index should be in [0, " +
		     to_str(this->val_real_(i).GetM()-1) + "], but is equal to "
		     + to_str(j) + ".");
#endif

    return val_real_(i).Value(j);
  }


  //! Returns column number of j-th non-zero value of row i (real part).
  /*!
    \param[in] i row number.
    \param[in] j local number.
    \return Column number of j-th non-zero entry of row i.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline int Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::
  IndexReal(int i, int j) const
  {

#ifdef SELDON_CHECK_BOUNDS
    if (i < 0 || i >= this->m_)
      throw WrongRow("Matrix_ArraySparse::index", "Index should be in [0, "
		     + to_str(this->m_-1) + "], but is equal to "
		     + to_str(i) + ".");

    if (j < 0 || j >= this->val_real_(i).GetM())
      throw WrongCol("Matrix_ArraySparse::index", "Index should be in [0, " +
		     to_str(this->val_real_(i).GetM()-1) + "], but is equal to "
		     + to_str(j) + ".");
#endif

    return val_real_(i).Index(j);
  }


  //! Returns column number of j-th non-zero value of row i (real part).
  /*!
    \param[in] i row number.
    \param[in] j local number.
    \return column number of j-th non-zero entry of row i.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline int& Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::
  IndexReal(int i, int j)
  {

#ifdef SELDON_CHECK_BOUNDS
    if (i < 0 || i >= this->m_)
      throw WrongRow("Matrix_ArraySparse::index", "Index should be in [0, "
		     + to_str(this->m_-1) + "], but is equal to "
		     + to_str(i) + ".");

    if (j < 0 || j >= this->val_real_(i).GetM())
      throw WrongCol("Matrix_ArraySparse::index", "Index should be in [0, "
		     + to_str(this->val_real_(i).GetM()-1)
		     + "], but is equal to " + to_str(j) + ".");
#endif

    return val_real_(i).Index(j);
  }


  //! Returns j-th non-zero value of row i (imaginary part).
  /*!
    \param[in] i row number.
    \param[in] j local number.
    \return j-th non-zero entry of row i.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline const T& Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::
  ValueImag(int i, int j) const
  {

#ifdef SELDON_CHECK_BOUNDS
    if (i < 0 || i >= this->m_)
      throw WrongRow("Matrix_ArraySparse::value", "Index should be in [0, "
		     + to_str(this->m_-1) + "], but is equal to "
		     + to_str(i) + ".");

    if (j < 0 || j >= this->val_imag_(i).GetM())
      throw WrongCol("Matrix_ArraySparse::value", "Index should be in [0, " +
		     to_str(this->val_imag_(i).GetM()-1) + "], but is equal to "
		     + to_str(j) + ".");
#endif

    return val_imag_(i).Value(j);
  }


  //! Returns j-th non-zero value of row i (imaginary part).
  /*!
    \param[in] i row number.
    \param[in] j local number.
    \return j-th non-zero entry of row i.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline T& Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::
  ValueImag (int i, int j)
  {

#ifdef SELDON_CHECK_BOUNDS
    if (i < 0 || i >= this->m_)
      throw WrongRow("Matrix_ArraySparse::value", "Index should be in [0, "
		     + to_str(this->m_-1) + "], but is equal to "
		     + to_str(i) + ".");

    if (j < 0 || j >= this->val_imag_(i).GetM())
      throw WrongCol("Matrix_ArraySparse::value", "Index should be in [0, " +
		     to_str(this->val_imag_(i).GetM()-1) + "], but is equal to "
		     + to_str(j) + ".");
#endif

    return val_imag_(i).Value(j);
  }


  //! Returns column number of j-th non-zero value of row i (imaginary part).
  /*!
    \param[in] i row number.
    \param[in] j local number.
    \return Column number of j-th non-zero entry of row i.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline int Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::
  IndexImag(int i, int j) const
  {

#ifdef SELDON_CHECK_BOUNDS
    if (i < 0 || i >= this->m_)
      throw WrongRow("Matrix_ArraySparse::index", "Index should be in [0, "
		     + to_str(this->m_-1) + "], but is equal to "
		     + to_str(i) + ".");

    if (j < 0 || j >= this->val_imag_(i).GetM())
      throw WrongCol("Matrix_ArraySparse::index", "Index should be in [0, " +
		     to_str(this->val_imag_(i).GetM()-1) + "], but is equal to "
		     + to_str(j) + ".");
#endif

    return val_imag_(i).Index(j);
  }


  //! Returns column number of j-th non-zero value of row i (imaginary part).
  /*!
    \param[in] i row number.
    \param[in] j local number.
    \return column number of j-th non-zero entry of row i.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline int& Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::
  IndexImag(int i, int j)
  {

#ifdef SELDON_CHECK_BOUNDS
    if (i < 0 || i >= this->m_)
      throw WrongRow("Matrix_ArrayComplexSparse::index",
		     "Index should be in [0, " + to_str(this->m_-1)
		     + "], but is equal to " + to_str(i) + ".");

    if (j < 0 || j >= this->val_imag_(i).GetM())
      throw WrongCol("Matrix_ArraySparse::index", "Index should be in [0, " +
		     to_str(this->val_imag_(i).GetM()-1) + "], but is equal to "
		     + to_str(j) + ".");
#endif

    return val_imag_(i).Index(j);
  }


  //! Redefines a row/column of the matrix
  /*!
    \param[in] i row/col number
    \param[in] n number of non-zero entries in the row
    \param[in] val values
    \param[in] ind column numbers
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline void Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::
  SetRealData(int i, int n, T* val, int* ind)
  {
    val_real_(i).SetData(n, val, ind);
  }


  //! Redefines a row/column of the matrix
  /*!
    \param[in] i row/col number
    \param[in] n number of non-zero entries in the row
    \param[in] val values
    \param[in] ind column numbers
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline void Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::
  SetImagData(int i, int n, T* val, int* ind)
  {
    val_imag_(i).SetData(n, val, ind);
  }


  //!  Clears a row without releasing memory.
  /*!
    On exit, the row is empty and the memory has not been released.
    It is useful for low level manipulations on a Matrix instance.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline void Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::NullifyReal(int i)
  {
    val_real_(i).Nullify();
  }


  //!  Clears a row without releasing memory.
  /*!
    On exit, the row is empty and the memory has not been released.
    It is useful for low level manipulations on a Matrix instance.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline void Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::NullifyImag(int i)
  {
    val_imag_(i).Nullify();
  }


  //! Redefines the real part of the matrix.
  /*!
    \param[in] m new number of rows.
    \param[in] n new number of columns.
    \param[in] val array of sparse rows/columns.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline void Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::
  SetRealData(int m, int n, Vector<T, VectSparse, Allocator>* val)
  {
    m_ = m;
    n_ = n;
    val_real_.SetData(Storage::GetFirst(m, n), val);
  }


  //! Redefines the imaginary part of the matrix.
  /*!
    \param[in] m new number of rows.
    \param[in] n new number of columns.
    \param[in] val array of sparse rows/columns.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline void Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::
  SetImagData(int m, int n, Vector<T, VectSparse, Allocator>* val)
  {
    m_ = m;
    n_ = n;
    val_imag_.SetData(Storage::GetFirst(m, n), val);
  }


  //!  Clears the matrix without releasing memory.
  /*!
    On exit, the matrix is empty and the memory has not been released.
    It is useful for low level manipulations on a Matrix instance.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline void Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::NullifyReal()
  {
    m_ = 0;
    n_ = 0;
    val_real_.Nullify();
  }


  //!  Clears the matrix without releasing memory.
  /*!
    On exit, the matrix is empty and the memory has not been released.
    It is useful for low level manipulations on a Matrix instance.
  */
  template <class T, class Prop, class Storage, class Allocator>
  inline void Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::NullifyImag()
  {
    m_ = 0;
    n_ = 0;
    val_imag_.Nullify();
  }


  /************************
   * CONVENIENT FUNCTIONS *
   ************************/


  //! Displays the matrix on the standard output.
  /*!
    Displays elements on the standard output, in text format.
    Each row is displayed on a single line and elements of
    a row are delimited by tabulations.
  */
  template <class T, class Prop, class Storage, class Allocator>
  void Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::Print() const
  {
    if (Storage::GetFirst(1, 0) == 1)
      for (int i = 0; i < this->m_; i++)
	{
	  for (int j = 0; j < this->val_real_(i).GetM(); j++)
	    cout << (i+1) << " " << this->val_real_(i).Index(j)+1
		 << " " << this->val_real_(i).Value(j) << endl;

	  for (int j = 0; j < this->val_imag_(i).GetM(); j++)
	    cout << (i+1) << " " << this->val_imag_(i).Index(j)+1
		 << " (0, " << this->val_imag_(i).Value(j) << ")"<<endl;
	}
    else
      for (int i = 0; i < this->n_; i++)
	{
	  for (int j = 0; j < this->val_real_(i).GetM(); j++)
	    cout << this->val_real_(i).Index(j)+1 << " " << i+1
		 << " " << this->val_real_(i).Value(j) << endl;

	  for (int j = 0; j < this->val_imag_(i).GetM(); j++)
	    cout << this->val_imag_(i).Index(j)+1 << " " << i+1
		 << " (0, " << this->val_imag_(i).Value(j) << ")"<<endl;
	}
  }


  //! Assembles the matrix.
  /*!
    All the row numbers are sorted.
    If same row numbers exist, values are added.
    \warning If you are using the methods AddInteraction/AddInteractions,
    you don't need to call that method.
  */
  template <class T, class Prop, class Storage, class Allocator>
  void Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::Assemble()
  {
    for (int i = 0; i < m_ ; i++)
      {
	val_real_(i).Assemble();
	val_imag_(i).Assemble();
      }
  }


  //! Matrix is initialized to the identity matrix.
  template <class T, class Prop, class Storage, class Allocator>
  inline void Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::
  SetIdentity()
  {
    this->n_ = this->m_;
    for (int i = 0; i < this->m_; i++)
      {
	val_real_(i).Reallocate(1);
	val_real_(i).Index(0) = i;
	val_real_(i).Value(0) = T(1);
      }
  }


  //! Non-zero entries are set to 0 (but not removed).
  template <class T, class Prop, class Storage, class Allocator>
  inline void Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::Zero()
  {
    for (int i = 0; i < this->m_; i++)
      {
	val_real_(i).Zero();
	val_imag_(i).Zero();
      }
  }


  //! Non-zero entries are filled with values 0, 1, 2, 3 ...
  template <class T, class Prop, class Storage, class Allocator>
  inline void Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::Fill()
  {
    int value = 0;
    for (int i = 0; i < this->m_; i++)
      {
	for (int j = 0; j < val_real_(i).GetM(); j++)
	  val_real_(i).Value(j) = value++;

	for (int j = 0; j < val_imag_(i).GetM(); j++)
	  val_imag_(i).Value(j) = value++;
      }
  }


  //! Non-zero entries are set to a given value x.
  template <class T, class Prop, class Storage, class Allo> template<class T0>
  inline void Matrix_ArrayComplexSparse<T, Prop, Storage, Allo>::
  Fill(const complex<T0>& x)
  {
    for (int i = 0; i < this->m_; i++)
      {
	val_real_(i).Fill(real(x));
	val_imag_(i).Fill(imag(x));
      }
  }


  //! Non-zero entries are set to a given value x.
  template <class T, class Prop, class Storage, class Allocator>
  template <class T0>
  inline Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>&
  Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::operator=
  (const complex<T0>& x)
  {
    this->Fill(x);
  }


  //! Non-zero entries take a random value.
  template <class T, class Prop, class Storage, class Allocator>
  inline void Matrix_ArrayComplexSparse<T, Prop, Storage, Allocator>::
  FillRand()
  {
    for (int i = 0; i < this->m_; i++)
      {
	val_real_(i).FillRand();
	val_imag_(i).FillRand();
      }
  }


  ////////////////////////////////////
  // MATRIX<ARRAY_COLCOMPLEXSPARSE> //
  ////////////////////////////////////


  //! Default constructor.
  /*!
    Builds an empty matrix.
  */
  template <class T, class Prop, class Allocator>
  inline Matrix<T, Prop, ArrayColComplexSparse, Allocator>::Matrix()  throw():
    Matrix_ArrayComplexSparse<T, Prop, ArrayColComplexSparse, Allocator>()
  {
  }


  //! Constructor.
  /*! Builds a i by j matrix.
    \param i number of rows.
    \param j number of columns.
    \note Matrix values are not initialized.
  */
  template <class T, class Prop, class Allocator>
  inline Matrix<T, Prop, ArrayColComplexSparse, Allocator>::Matrix(int i, int j):
    Matrix_ArrayComplexSparse<T, Prop, ArrayColComplexSparse, Allocator>(i, j)
  {
  }


  //! Clears column i.
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayColComplexSparse, Allocator>::ClearRealColumn(int i)
  {
    this->val_real_(i).Clear();
  }


  //! Clears column i.
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayColComplexSparse, Allocator>::ClearImagColumn(int i)
  {
    this->val_imag_(i).Clear();
  }


  //! Reallocates column i.
  /*!
    \param[in] i column number.
    \param[in] j new number of non-zero entries in the column.
  */
  template <class T, class Prop, class Alloc> inline
  void Matrix<T, Prop, ArrayColComplexSparse, Alloc>::ReallocateRealColumn(int i,int j)
  {
    this->val_real_(i).Reallocate(j);
  }


  //! Reallocates column i.
  /*!
    \param[in] i column number.
    \param[in] j new number of non-zero entries in the column.
  */
  template <class T, class Prop, class Alloc> inline
  void Matrix<T, Prop, ArrayColComplexSparse, Alloc>::ReallocateImagColumn(int i,int j)
  {
    this->val_imag_(i).Reallocate(j);
  }


  //! Reallocates column i.
  /*!
    \param[in] i column number.
    \param[in] j new number of non-zero entries in the column.
  */
  template <class T, class Prop, class Allocator> inline
  void Matrix<T, Prop, ArrayColComplexSparse, Allocator>::ResizeRealColumn(int i,int j)
  {
    this->val_real_(i).Resize(j);
  }


  //! Reallocates column i.
  /*!
    \param[in] i column number.
    \param[in] j new number of non-zero entries in the column.
  */
  template <class T, class Prop, class Allocator> inline
  void Matrix<T, Prop, ArrayColComplexSparse, Allocator>::ResizeImagColumn(int i,int j)
  {
    this->val_imag_(i).Resize(j);
  }


  //! Swaps two columns.
  /*!
    \param[in] i first column number.
    \param[in] j second column number.
  */
  template <class T, class Prop, class Allocator> inline
  void Matrix<T, Prop, ArrayColComplexSparse, Allocator>::SwapRealColumn(int i,int j)
  {
    Swap(this->val_real_(i), this->val_real_(j));
  }


  //! Swaps two columns.
  /*!
    \param[in] i first column number.
    \param[in] j second column number.
  */
  template <class T, class Prop, class Allocator> inline
  void Matrix<T, Prop, ArrayColComplexSparse, Allocator>::SwapImagColumn(int i,int j)
  {
    Swap(this->val_imag_(i), this->val_imag_(j));
  }


  //! Sets row numbers of non-zero entries of a column.
  /*!
    \param[in] i column number.
    \param[in] new_index new row numbers.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayColComplexSparse, Allocator>::
  ReplaceRealIndexColumn(int i, IVect& new_index)
  {
    for (int j = 0; j < this->val_real_(i).GetM(); j++)
      this->val_real_(i).Index(j) = new_index(j);
  }


  //! Sets row numbers of non-zero entries of a column.
  /*!
    \param[in] i column number.
    \param[in] new_index new row numbers.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayColComplexSparse, Allocator>::
  ReplaceImagIndexColumn(int i, IVect& new_index)
  {
    for (int j = 0; j < this->val_imag_(i).GetM(); j++)
      this->val_imag_(i).Index(j) = new_index(j);
  }


  //! Returns the number of non-zero entries of a column.
  /*!
    \param[in] i column number.
    \return The number of non-zero entries of the column i.
  */
  template <class T, class Prop, class Allocator>
  inline int Matrix<T, Prop, ArrayColComplexSparse, Allocator>::
  GetRealColumnSize(int i) const
  {
    return this->val_real_(i).GetSize();
  }


  //! Returns the number of non-zero entries of a column.
  /*!
    \param[in] i column number.
    \return The number of non-zero entries of the column i.
  */
  template <class T, class Prop, class Allocator>
  inline int Matrix<T, Prop, ArrayColComplexSparse, Allocator>::
  GetImagColumnSize(int i) const
  {
    return this->val_imag_(i).GetSize();
  }


  //! Displays non-zero values of a column.
  template <class T, class Prop, class Allocator> inline
  void Matrix<T, Prop, ArrayColComplexSparse, Allocator>::PrintRealColumn(int i) const
  {
    this->val_real_(i).Print();
  }


  //! Displays non-zero values of a column.
  template <class T, class Prop, class Allocator> inline
  void Matrix<T, Prop, ArrayColComplexSparse, Allocator>::PrintImagColumn(int i) const
  {
    this->val_imag_(i).Print();
  }


  //! Assembles a column.
  /*!
    \param[in] i column number.
    \warning If you are using the methods AddInteraction,
    you don't need to call that method.
  */
  template <class T, class Prop, class Allocator> inline
  void Matrix<T, Prop, ArrayColComplexSparse, Allocator>::AssembleRealColumn(int i)
  {
    this->val_real_(i).Assemble();
  }


  //! Assembles a column.
  /*!
    \param[in] i column number.
    \warning If you are using the methods AddInteraction,
    you don't need to call that method.
  */
  template <class T, class Prop, class Allocator> inline
  void Matrix<T, Prop, ArrayColComplexSparse, Allocator>::AssembleImagColumn(int i)
  {
    this->val_imag_(i).Assemble();
  }


  //! Adds a coefficient in the matrix.
  /*!
    \param[in] i row number.
    \param[in] j column number.
    \param[in] val coefficient to add.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayColComplexSparse, Allocator>::
  AddInteraction(int i, int j, const complex<T>& val)
  {
    if (real(val) != T(0))
      this->val_real_(j).AddInteraction(i, real(val));

    if (imag(val) != T(0))
      this->val_imag_(j).AddInteraction(i, imag(val));
  }


  //! Adds coefficients in a row.
  /*!
    \param[in] i row number.
    \param[in] nb number of coefficients to add.
    \param[in] col column numbers of coefficients.
    \param[in] val values of coefficients.
  */
  template <class T, class Prop, class Allocator> template <class Alloc1>
  inline void Matrix<T, Prop, ArrayColComplexSparse, Allocator>::
  AddInteractionRow(int i, int nb, const IVect& col,
		    const Vector<complex<T>, VectFull, Alloc1>& val)
  {
    for (int j = 0; j < nb; j++)
      AddInteraction(i, col(j), val(j));
  }


  //! Adds coefficients in a column.
  /*!
    \param[in] i column number.
    \param[in] nb number of coefficients to add.
    \param[in] row row numbers of coefficients.
    \param[in] val values of coefficients.
  */
  template <class T, class Prop, class Allocator> template <class Alloc1>
  inline void Matrix<T, Prop, ArrayColComplexSparse, Allocator>::
  AddInteractionColumn(int i, int nb, const IVect& row,
		       const Vector<complex<T>, VectFull, Alloc1>& val)
  {
    int nb_real = 0;
    int nb_imag = 0;
    IVect row_real(nb), row_imag(nb);
    Vector<T> val_real(nb), val_imag(nb);
    for (int j = 0; j < nb; j++)
      {
	if (real(val(j)) != T(0))
	  {
	    row_real(nb_real) = row(j);
	    val_real(nb_real) = real(val(j));
	    nb_real++;
	  }

	if (imag(val(j)) != T(0))
	  {
	    row_imag(nb_imag) = row(j);
	    val_imag(nb_imag) = imag(val(j));
	    nb_imag++;
	  }
      }

    this->val_real_(i).AddInteractionRow(nb_real, row_real, val_real);
    this->val_imag_(i).AddInteractionRow(nb_imag, row_imag, val_imag);
  }


  ////////////////////////////////////
  // MATRIX<ARRAY_ROWCOMPLEXSPARSE> //
  ////////////////////////////////////


  //! Default constructor.
  /*!
    Builds a empty matrix.
  */
  template <class T, class Prop, class Allocator>
  inline Matrix<T, Prop, ArrayRowComplexSparse, Allocator>::Matrix()  throw():
    Matrix_ArrayComplexSparse<T, Prop, ArrayRowComplexSparse, Allocator>()
  {
  }


  //! Constructor.
  /*! Builds a i by j matrix
    \param i number of rows.
    \param j number of columns.
    \note Matrix values are not initialized.
  */
  template <class T, class Prop, class Allocator>
  inline Matrix<T, Prop, ArrayRowComplexSparse, Allocator>::Matrix(int i, int j):
    Matrix_ArrayComplexSparse<T, Prop, ArrayRowComplexSparse, Allocator>(i, j)
  {
  }


  //! Clears a row
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayRowComplexSparse, Allocator>::ClearRealRow(int i)
  {
    this->val_real_(i).Clear();
  }


  //! Clears a row
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayRowComplexSparse, Allocator>::ClearImagRow(int i)
  {
    this->val_imag_(i).Clear();
  }


  //! Changes the size of a row.
  /*!
    \param[in] i row number.
    \param[in] j new number of non-zero entries of the row.
    \warning Data may be lost.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayRowComplexSparse, Allocator>::
  ReallocateRealRow(int i, int j)
  {
    this->val_real_(i).Reallocate(j);
  }


  //! Changes the size of a row.
  /*!
    \param[in] i row number.
    \param[in] j new number of non-zero entries of the row.
    \warning Data may be lost.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayRowComplexSparse, Allocator>::
  ReallocateImagRow(int i, int j)
  {
    this->val_imag_(i).Reallocate(j);
  }


  //! Changes the size of a row.
  /*!
    \param[in] i row number.
    \param[in] j new number of non-zero entries of the row.
    \note Data is kept.
  */
  template <class T, class Prop, class Allocator> inline
  void Matrix<T, Prop, ArrayRowComplexSparse, Allocator>::ResizeRealRow(int i, int j)
  {
    this->val_real_(i).Resize(j);
  }


  //! Changes the size of a row.
  /*!
    \param[in] i row number.
    \param[in] j new number of non-zero entries of the row.
    \note Data is kept.
  */
  template <class T, class Prop, class Allocator> inline
  void Matrix<T, Prop, ArrayRowComplexSparse, Allocator>::ResizeImagRow(int i, int j)
  {
    this->val_imag_(i).Resize(j);
  }


  //! Swaps two rows
  /*!
    \param[in] i first row number.
    \param[in] j second row number.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayRowComplexSparse, Allocator>::SwapRealRow(int i,int j)
  {
    Swap(this->val_real_(i), this->val_real_(j));
  }


  //! Swaps two rows
  /*!
    \param[in] i first row number.
    \param[in] j second row number.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayRowComplexSparse, Allocator>::SwapImagRow(int i,int j)
  {
    Swap(this->val_imag_(i), this->val_imag_(j));
  }


  //! Sets column numbers of non-zero entries of a row.
  /*!
    \param[in] i column number.
    \param[in] new_index new column numbers.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayRowComplexSparse, Allocator>::
  ReplaceRealIndexRow(int i, IVect& new_index)
  {
    for (int j = 0; j < this->val_real_(i).GetM(); j++)
      this->val_real_(i).Index(j) = new_index(j);
  }


  //! Sets column numbers of non-zero entries of a row.
  /*!
    \param[in] i column number.
    \param[in] new_index new column numbers.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayRowComplexSparse, Allocator>::
  ReplaceImagIndexRow(int i, IVect& new_index)
  {
    for (int j = 0; j < this->val_imag_(i).GetM(); j++)
      this->val_imag_(i).Index(j) = new_index(j);
  }


  //! Returns the number of non-zero entries of a row.
  /*!
    \param[in] i row number.
    \return The number of non-zero entries of the row i.
  */
  template <class T, class Prop, class Allocator> inline
  int Matrix<T, Prop, ArrayRowComplexSparse, Allocator>::GetRealRowSize(int i) const
  {
    return this->val_real_(i).GetSize();
  }


  //! Returns the number of non-zero entries of a row.
  /*!
    \param[in] i row number.
    \return The number of non-zero entries of the row i.
  */
  template <class T, class Prop, class Allocator> inline
  int Matrix<T, Prop, ArrayRowComplexSparse, Allocator>::GetImagRowSize(int i) const
  {
    return this->val_imag_(i).GetSize();
  }


  //! Displays non-zero values of a row.
  template <class T, class Prop, class Allocator> inline
  void Matrix<T, Prop, ArrayRowComplexSparse, Allocator>::PrintRealRow(int i) const
  {
    this->val_real_(i).Print();
  }


  //! Displays non-zero values of a row.
  template <class T, class Prop, class Allocator> inline
  void Matrix<T, Prop, ArrayRowComplexSparse, Allocator>::PrintImagRow(int i) const
  {
    this->val_imag_(i).Print();
  }


  //! Assembles a row.
  /*!
    \param[in] i row number.
    \warning If you are using the methods AddInteraction,
    you don't need to call that method.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayRowComplexSparse, Allocator>::AssembleRealRow(int i)
  {
    this->val_real_(i).Assemble();
  }


  //! Assembles a row.
  /*!
    \param[in] i row number.
    \warning If you are using the methods AddInteraction,
    you don't need to call that method.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayRowComplexSparse, Allocator>::AssembleImagRow(int i)
  {
    this->val_imag_(i).Assemble();
  }


  //! Adds a coefficient in the matrix.
  /*!
    \param[in] i row number.
    \param[in] j column number.
    \param[in] val coefficient to add.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayRowComplexSparse, Allocator>::
  AddInteraction(int i, int j, const complex<T>& val)
  {
    if (real(val) != T(0))
      this->val_real_(i).AddInteraction(j, real(val));

    if (imag(val) != T(0))
      this->val_imag_(i).AddInteraction(j, imag(val));
  }


  //! Adds coefficients in a row.
  /*!
    \param[in] i row number.
    \param[in] nb number of coefficients to add.
    \param[in] col column numbers of coefficients.
    \param[in] val values of coefficients.
  */
  template <class T, class Prop, class Allocator> template <class Alloc1>
  inline void Matrix<T, Prop, ArrayRowComplexSparse, Allocator>::
  AddInteractionRow(int i, int nb, const IVect& col,
		    const Vector<complex<T>, VectFull, Alloc1>& val)
  {
    int nb_real = 0;
    int nb_imag = 0;
    IVect col_real(nb), col_imag(nb);
    Vector<T> val_real(nb), val_imag(nb);
    for (int j = 0; j < nb; j++)
      {
	if (real(val(j)) != T(0))
	  {
	    col_real(nb_real) = col(j);
	    val_real(nb_real) = real(val(j));
	    nb_real++;
	  }

	if (imag(val(i)) != T(0))
	  {
	    col_imag(nb_imag) = col(j);
	    val_imag(nb_imag) = imag(val(j));
	    nb_imag++;
	  }
      }

    this->val_real_(i).AddInteractionRow(nb_real, col_real, val_real);
    this->val_imag_(i).AddInteractionRow(nb_imag, col_imag, val_imag);
  }


  //! Adds coefficients in a column.
  /*!
    \param[in] i column number.
    \param[in] nb number of coefficients to add.
    \param[in] row row numbers of coefficients.
    \param[in] val values of coefficients.
  */
  template <class T, class Prop, class Allocator> template <class Alloc1>
  inline void Matrix<T, Prop, ArrayRowComplexSparse, Allocator>::
  AddInteractionColumn(int i, int nb, const IVect& row,
		       const Vector<complex<T>, VectFull, Alloc1>& val)
  {
    for (int j = 0; j < nb; j++)
      AddInteraction(row(j), i, val(j));
  }


  ///////////////////////////////////////
  // MATRIX<ARRAY_COLSYMCOMPLEXSPARSE> //
  ///////////////////////////////////////


  //! Default constructor.
  /*!
    Builds a empty matrix.
  */
  template <class T, class Prop, class Allocator>
  inline Matrix<T, Prop, ArrayColSymComplexSparse, Allocator>::Matrix()  throw():
    Matrix_ArrayComplexSparse<T, Prop, ArrayColSymComplexSparse, Allocator>()
  {
  }


  //! Constructor.
  /*! Builds a i by j matrix
    \param i number of rows.
    \param j number of columns.
    \note Matrix values are not initialized.
  */
  template <class T, class Prop, class Allocator>
  inline Matrix<T, Prop, ArrayColSymComplexSparse, Allocator>::Matrix(int i, int j):
    Matrix_ArrayComplexSparse<T, Prop, ArrayColSymComplexSparse, Allocator>(i, j)
  {
  }


  /**********************************
   * ELEMENT ACCESS AND AFFECTATION *
   **********************************/


  //! Access operator.
  /*!
    Returns the value of element (i, j).
    \param i row index.
    \param j column index.
    \return Element (i, j) of the matrix.
  */
  template <class T, class Prop, class Allocator>
  inline complex<T>
  Matrix<T, Prop, ArrayColSymComplexSparse, Allocator>::operator() (int i, int j)
    const
  {

#ifdef SELDON_CHECK_BOUNDS
    if (i < 0 || i >= this->m_)
      throw WrongRow("Matrix::operator()", "Index should be in [0, "
		     + to_str(this->m_-1) + "], but is equal to "
		     + to_str(i) + ".");
    if (j < 0 || j >= this->n_)
      throw WrongCol("Matrix::operator()", "Index should be in [0, "
		     + to_str(this->n_-1) + "], but is equal to "
		     + to_str(j) + ".");
#endif

    if (i <= j)
      return complex<T>(this->val_real_(j)(i), this->val_imag_(j)(i));

    return complex<T>(this->val_real_(i)(j), this->val_imag_(i)(j));
  }


  //! Clears a column.
  template <class T, class Prop, class Allocator> inline
  void Matrix<T, Prop, ArrayColSymComplexSparse, Allocator>::ClearRealColumn(int i)
  {
    this->val_real_(i).Clear();
  }


  //! Clears a column.
  template <class T, class Prop, class Allocator> inline
  void Matrix<T, Prop, ArrayColSymComplexSparse, Allocator>::ClearImagColumn(int i)
  {
    this->val_imag_(i).Clear();
  }


  //! Reallocates column i.
  /*!
    \param[in] i column number.
    \param[in] j new number of non-zero entries in the column.
    \warning Data may be lost.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayColSymComplexSparse, Allocator>::
  ReallocateRealColumn(int i, int j)
  {
    this->val_real_(i).Reallocate(j);
  }


  //! Reallocates column i.
  /*!
    \param[in] i column number.
    \param[in] j new number of non-zero entries in the column.
    \warning Data may be lost.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayColSymComplexSparse, Allocator>::
  ReallocateImagColumn(int i, int j)
  {
    this->val_imag_(i).Reallocate(j);
  }


  //! Reallocates column i.
  /*!
    \param[in] i column number.
    \param[in] j new number of non-zero entries in the column.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayColSymComplexSparse, Allocator>::
  ResizeRealColumn(int i, int j)
  {
    this->val_real_(i).Resize(j);
  }


  //! Reallocates column i.
  /*!
    \param[in] i column number.
    \param[in] j new number of non-zero entries in the column.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayColSymComplexSparse, Allocator>::
  ResizeImagColumn(int i, int j)
  {
    this->val_imag_(i).Resize(j);
  }


  //! Swaps two columns.
  /*!
    \param[in] i first column number.
    \param[in] j second column number.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayColSymComplexSparse, Allocator>::
  SwapRealColumn(int i, int j)
  {
    Swap(this->val_real_(i), this->val_real_(j));
  }


  //! Swaps two columns.
  /*!
    \param[in] i first column number.
    \param[in] j second column number.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayColSymComplexSparse, Allocator>::
  SwapImagColumn(int i, int j)
  {
    Swap(this->val_imag_(i), this->val_imag_(j));
  }


  //! Sets row numbers of non-zero entries of a column.
  /*!
    \param[in] i column number.
    \param[in] new_index new row numbers.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayColSymComplexSparse, Allocator>::
  ReplaceRealIndexColumn(int i, IVect& new_index)
  {
    for (int j = 0; j < this->val_real_(i).GetM(); j++)
      this->val_real_(i).Index(j) = new_index(j);
  }


  //! Sets row numbers of non-zero entries of a column.
  /*!
    \param[in] i column number.
    \param[in] new_index new row numbers.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayColSymComplexSparse, Allocator>::
  ReplaceImagIndexColumn(int i, IVect& new_index)
  {
    for (int j = 0; j < this->val_imag_(i).GetM(); j++)
      this->val_imag_(i).Index(j) = new_index(j);
  }


  //! Returns the number of non-zero entries of a column.
  /*!
    \param[in] i column number.
    \return The number of non-zero entries of the column i.
  */
  template <class T, class Prop, class Allocator>
  inline int Matrix<T, Prop, ArrayColSymComplexSparse, Allocator>::
  GetRealColumnSize(int i) const
  {
    return this->val_real_(i).GetSize();
  }


  //! Returns the number of non-zero entries of a column.
  /*!
    \param[in] i column number.
    \return The number of non-zero entries of the column i.
  */
  template <class T, class Prop, class Allocator>
  inline int Matrix<T, Prop, ArrayColSymComplexSparse, Allocator>::
  GetImagColumnSize(int i) const
  {
    return this->val_imag_(i).GetSize();
  }


  //! Displays non-zero values of a column.
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayColSymComplexSparse, Allocator>::
  PrintRealColumn(int i) const
  {
    this->val_real_(i).Print();
  }


  //! Displays non-zero values of a column.
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayColSymComplexSparse, Allocator>::
  PrintImagColumn(int i) const
  {
    this->val_imag_(i).Print();
  }


  //! Assembles a column.
  /*!
    \param[in] i column number.
    \warning If you are using the methods AddInteraction,
    you don't need to call that method.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayColSymComplexSparse, Allocator>::
  AssembleRealColumn(int i)
  {
    this->val_real_(i).Assemble();
  }


  //! Assembles a column.
  /*!
    \param[in] i column number.
    \warning If you are using the methods AddInteraction,
    you don't need to call that method.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayColSymComplexSparse, Allocator>::
  AssembleImagColumn(int i)
  {
    this->val_imag_(i).Assemble();
  }


  //! Adds a coefficient in the matrix.
  /*!
    \param[in] i row number.
    \param[in] j column number.
    \param[in] val coefficient to add.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayColSymComplexSparse, Allocator>::
  AddInteraction(int i, int j, const complex<T>& val)
  {
    if (i <= j)
      {
	if (real(val) != T(0))
	  this->val_real_(j).AddInteraction(i, real(val));

	if (imag(val) != T(0))
	  this->val_imag_(j).AddInteraction(i, imag(val));
      }
  }


  //! Adds coefficients in a row.
  /*!
    \param[in] i row number.
    \param[in] nb number of coefficients to add.
    \param[in] col column numbers of coefficients.
    \param[in] val values of coefficients.
  */
  template <class T, class Prop, class Allocator> template <class Alloc1>
  inline void Matrix<T, Prop, ArrayColSymComplexSparse, Allocator>::
  AddInteractionRow(int i, int nb, const IVect& col,
		    const Vector<complex<T>, VectFull, Alloc1>& val)
  {
    AddInteractionColumn(i, nb, col, val);
  }


  //! Adds coefficients in a column.
  /*!
    \param[in] i column number.
    \param[in] nb number of coefficients to add.
    \param[in] row row numbers of coefficients.
    \param[in] val values of coefficients.
  */
  template <class T, class Prop, class Allocator> template <class Alloc1>
  inline void Matrix<T, Prop, ArrayColSymComplexSparse, Allocator>::
  AddInteractionColumn(int i, int nb, const IVect& row,
		       const Vector<complex<T>, VectFull, Alloc1>& val)
  {
    int nb_real = 0;
    int nb_imag = 0;
    IVect row_real(nb), row_imag(nb);
    Vector<T> val_real(nb), val_imag(nb);
    for (int j = 0; j < nb; j++)
      if (row(j) <= i)
	{
	  if (real(val(j)) != T(0))
	    {
	      row_real(nb_real) = row(j);
	      val_real(nb_real) = real(val(j));
	      nb_real++;
	    }

	  if (imag(val(j)) != T(0))
	    {
	      row_imag(nb_imag) = row(j);
	      val_imag(nb_imag) = imag(val(j));
	      nb_imag++;
	    }
	}

    this->val_real_(i).AddInteractionRow(nb_real, row_real, val_real);
    this->val_imag_(i).AddInteractionRow(nb_imag, row_imag, val_imag);
  }


  ///////////////////////////////////////
  // MATRIX<ARRAY_ROWSYMCOMPLEXSPARSE> //
  ///////////////////////////////////////


  //! Default constructor.
  /*!
    Builds a empty matrix.
  */
  template <class T, class Prop, class Allocator>
  inline Matrix<T, Prop, ArrayRowSymComplexSparse, Allocator>::Matrix()  throw():
    Matrix_ArrayComplexSparse<T, Prop, ArrayRowSymComplexSparse, Allocator>()
  {
  }


  //! Constructor.
  /*! Builds a i by j matrix
    \param i number of rows.
    \param j number of columns.
    \note Matrix values are not initialized.
  */
  template <class T, class Prop, class Allocator>
  inline Matrix<T, Prop, ArrayRowSymComplexSparse, Allocator>::Matrix(int i, int j):
    Matrix_ArrayComplexSparse<T, Prop, ArrayRowSymComplexSparse, Allocator>(i, j)
  {
  }


  /**********************************
   * ELEMENT ACCESS AND AFFECTATION *
   **********************************/


  //! Access operator.
  /*!
    Returns the value of element (i, j).
    \param i row index.
    \param j column index.
    \return Element (i, j) of the matrix.
  */
  template <class T, class Prop, class Allocator>
  inline complex<T>
  Matrix<T, Prop, ArrayRowSymComplexSparse, Allocator>::operator() (int i, int j)
    const
  {

#ifdef SELDON_CHECK_BOUNDS
    if (i < 0 || i >= this->m_)
      throw WrongRow("Matrix_ArraySparse::operator()", "Index should be in [0, "
		     + to_str(this->m_-1) + "], but is equal to "
		     + to_str(i) + ".");
    if (j < 0 || j >= this->n_)
      throw WrongCol("Matrix_ArraySparse::operator()", "Index should be in [0, "
		     + to_str(this->n_-1) + "], but is equal to "
		     + to_str(j) + ".");
#endif

    if (i <= j)
      return complex<T>(this->val_real_(i)(j), this->val_imag_(i)(j));

    return complex<T>(this->val_real_(j)(i), this->val_imag_(j)(i));
  }


  //! Clears a row.
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayRowSymComplexSparse, Allocator>::ClearRealRow(int i)
  {
    this->val_real_(i).Clear();
  }


  //! Clears a row.
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayRowSymComplexSparse, Allocator>::ClearImagRow(int i)
  {
    this->val_imag_(i).Clear();
  }


  //! Reallocates row i.
  /*!
    \param[in] i row number.
    \param[in] j new number of non-zero entries in the row.
    \warning Data may be lost.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayRowSymComplexSparse, Allocator>::
  ReallocateRealRow(int i,int j)
  {
    this->val_real_(i).Reallocate(j);
  }


  //! Reallocates row i.
  /*!
    \param[in] i row number.
    \param[in] j new number of non-zero entries in the row.
    \warning Data may be lost.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayRowSymComplexSparse, Allocator>::
  ReallocateImagRow(int i,int j)
  {
    this->val_imag_(i).Reallocate(j);
  }


  //! Reallocates row i.
  /*!
    \param[in] i column number.
    \param[in] j new number of non-zero entries in the row.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayRowSymComplexSparse, Allocator>::
  ResizeRealRow(int i,int j)
  {
    this->val_real_(i).Resize(j);
  }


  //! Reallocates row i.
  /*!
    \param[in] i column number.
    \param[in] j new number of non-zero entries in the row.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayRowSymComplexSparse, Allocator>::
  ResizeImagRow(int i,int j)
  {
    this->val_imag_(i).Resize(j);
  }


  //! Swaps two rows.
  /*!
    \param[in] i first row number.
    \param[in] j second row number.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayRowSymComplexSparse, Allocator>::
  SwapRealRow(int i,int j)
  {
    Swap(this->val_real_(i), this->val_real_(j));
  }


  //! Swaps two rows.
  /*!
    \param[in] i first row number.
    \param[in] j second row number.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayRowSymComplexSparse, Allocator>::
  SwapImagRow(int i,int j)
  {
    Swap(this->val_imag_(i), this->val_imag_(j));
  }


  //! Sets column numbers of non-zero entries of a row.
  /*!
    \param[in] i row number.
    \param[in] new_index new column numbers.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayRowSymComplexSparse, Allocator>::
  ReplaceRealIndexRow(int i, IVect& new_index)
  {
    for (int j = 0; j < this->val_real_(i).GetM(); j++)
      this->val_real_(i).Index(j) = new_index(j);
  }


  //! Sets column numbers of non-zero entries of a row.
  /*!
    \param[in] i row number.
    \param[in] new_index new column numbers.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayRowSymComplexSparse, Allocator>::
  ReplaceImagIndexRow(int i,IVect& new_index)
  {
    for (int j = 0; j < this->val_imag_(i).GetM(); j++)
      this->val_imag_(i).Index(j) = new_index(j);
  }


  //! Returns the number of non-zero entries of a row.
  /*!
    \param[in] i row number.
    \return The number of non-zero entries of the row i.
  */
  template <class T, class Prop, class Allocator>
  inline int Matrix<T, Prop, ArrayRowSymComplexSparse, Allocator>::GetRealRowSize(int i)
    const
  {
    return this->val_real_(i).GetSize();
  }


  //! Returns the number of non-zero entries of a row.
  /*!
    \param[in] i row number.
    \return The number of non-zero entries of the row i.
  */
  template <class T, class Prop, class Allocator>
  inline int Matrix<T, Prop, ArrayRowSymComplexSparse, Allocator>::GetImagRowSize(int i)
    const
  {
    return this->val_imag_(i).GetSize();
  }


  //! Displays non-zero values of a column.
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayRowSymComplexSparse, Allocator>::PrintRealRow(int i)
    const
  {
    this->val_real_(i).Print();
  }


  //! Displays non-zero values of a column.
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayRowSymComplexSparse, Allocator>::PrintImagRow(int i)
    const
  {
    this->val_imag_(i).Print();
  }


  //! Assembles a column.
  /*!
    \param[in] i column number.
    \warning If you are using the methods AddInteraction,
    you don't need to call that method.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayRowSymComplexSparse, Allocator>::AssembleRealRow(int i)
  {
    this->val_real_(i).Assemble();
  }


  //! Assembles a column.
  /*!
    \param[in] i column number.
    \warning If you are using the methods AddInteraction,
    you don't need to call that method.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayRowSymComplexSparse, Allocator>::AssembleImagRow(int i)
  {
    this->val_imag_(i).Assemble();
  }


  //! Adds a coefficient in the matrix.
  /*!
    \param[in] i row number.
    \param[in] j column number.
    \param[in] val coefficient to add.
  */
  template <class T, class Prop, class Allocator>
  inline void Matrix<T, Prop, ArrayRowSymComplexSparse, Allocator>::
  AddInteraction(int i, int j, const complex<T>& val)
  {
    if (i <= j)
      {
	if (real(val) != T(0))
	  this->val_real_(i).AddInteraction(j, real(val));

	if (imag(val) != T(0))
	  this->val_imag_(i).AddInteraction(j, imag(val));
      }
  }


  //! Adds coefficients in a row.
  /*!
    \param[in] i row number.
    \param[in] nb number of coefficients to add.
    \param[in] col column numbers of coefficients.
    \param[in] val values of coefficients.
  */
  template <class T, class Prop, class Allocator> template <class Alloc1>
  inline void Matrix<T, Prop, ArrayRowSymComplexSparse, Allocator>::
  AddInteractionRow(int i, int nb, const IVect& col,
		    const Vector<complex<T>, VectFull, Alloc1>& val)
  {
    int nb_real = 0;
    int nb_imag = 0;
    IVect col_real(nb), col_imag(nb);
    Vector<T> val_real(nb), val_imag(nb);
    for (int j = 0; j < nb; j++)
      if (i <= col(j))
	{
	  if (real(val(j)) != T(0))
	    {
	      col_real(nb_real) = col(j);
	      val_real(nb_real) = real(val(j));
	      nb_real++;
	    }

	  if (imag(val(j)) != T(0))
	    {
	      col_imag(nb_imag) = col(j);
	      val_imag(nb_imag) = imag(val(j));
	      nb_imag++;
	    }
	}

    this->val_real_(i).AddInteractionRow(nb_real, col_real, val_real);
    this->val_imag_(i).AddInteractionRow(nb_imag, col_imag, val_imag);
  }


  //! Adds coefficients in a column.
  /*!
    \param[in] i column number.
    \param[in] nb number of coefficients to add.
    \param[in] row row numbers of coefficients.
    \param[in] val values of coefficients.
  */
  template <class T, class Prop, class Allocator> template <class Alloc1>
  inline void Matrix<T, Prop, ArrayRowSymComplexSparse, Allocator>::
  AddInteractionColumn(int i, int nb, const IVect& row,
		       const Vector<complex<T>,VectFull,Alloc1>& val)
  {
    // Symmetric matrix, row = column.
    AddInteractionRow(i, nb, row, val);
  }

} // namespace Seldon

#define SELDON_FILE_MATRIX_ARRAY_COMPLEX_SPARSE_CXX
#endif
