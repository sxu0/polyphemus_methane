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


#ifndef SELDON_FILE_PERMUTATION_SCALING_MATRIX_CXX

/*
  Functions defined in this file:

  PermuteMatrix(A, I, J)

  ScaleMatrix(A, Drow, Dcol)

  ScaleLeftMatrix(A, Drow)
*/

namespace Seldon
{

  //! Permutation of a general matrix stored by rows.
  /*!
    B(row_perm(i), col_perm(j)) = A(i,j) and A = B.
    Equivalent Matlab operation: A(row_perm, col_perm) = A.
  */
  template<class T, class Prop, class Allocator>
  void PermuteMatrix(Matrix<T, Prop, ArrayRowSparse, Allocator>& A,
		     const IVect& row_perm, const IVect& col_perm)
  {
    int m = A.GetM(), n, i, i_, j, i2;
    IVect ind_tmp, iperm(m), rperm(m);
    for (i = 0; i < m; i++)
      {
	iperm(i) = i;
	rperm(i) = i;
      }
    // A(rperm(i),:) will be the place where is the initial row i.

    // Algorithm avoiding the allocation of another matrix.
    for (i = 0; i < m; i++)
      {
	// We get the index of row where the row initially placed on row i is.
	i2 = rperm(i);
	// We get the new index of this row.
	i_ = row_perm(i);

	// We fill ind_tmp of the permuted indices of columns of row i.
	n = A.GetRowSize(i2);
	ind_tmp.Reallocate(n);
	for (j = 0; j < n; j++)
	  ind_tmp(j) = col_perm(A.Index(i2,j));

	// We swap the two rows i and its destination row_perm(i).
	A.SwapRow(i2, i_);
	A.ReplaceIndexRow(i_, ind_tmp);

	// We update the indices iperm and rperm in order to keep in memory
	// the place where the row row_perm(i) is.
	int i_tmp = iperm(i_);
	iperm(i_) = iperm(i2);
	iperm(i2) = i_tmp;
	rperm(iperm(i_)) = i_;
	rperm(iperm(i2)) = i2;

	// We assemble the row i.
	A.AssembleRow(i_);
      }
  }


  //! Permutation of a symmetric matrix stored by columns.
  /*!
    B(row_perm(i),col_perm(j)) = A(i,j) and A = B.
    Equivalent Matlab operation: A(row_perm, col_perm) = A.
  */
  template<class T, class Prop, class Allocator>
  void PermuteMatrix(Matrix<T, Prop, ArrayColSymSparse, Allocator>& A,
		     const IVect& row_perm, const IVect& col_perm)
  {
    // It is assumed that the permuted matrix is still symmetric! For example,
    // the user can provide row_perm = col_perm.
    int m = A.GetM();
    int nnz = A.GetDataSize();
    IVect IndRow(nnz), IndCol(nnz);
    Vector<T, VectFull, Allocator> Val(nnz);

    // First we convert the matrix in coordinate format and we permute the
    // indices.
    // IndRow -> indices of the permuted rows
    // IndCol -> indices of the permuted columns
    int k = 0;
    for (int i = 0; i < m; i++)
      for (int j = 0; j < A.GetColumnSize(i); j++)
	{
	  IndCol(k) = col_perm(i);
	  Val(k) = A.Value(i,j);
	  IndRow(k) = row_perm(A.Index(i, j));
	  if (IndCol(k) <= IndRow(k))
	    {
	      // We store only the superior part of the symmetric matrix.
	      int ind_tmp = IndRow(k);
	      IndRow(k) = IndCol(k);
	      IndCol(k) = ind_tmp;
	    }
	  k++;
	}

    // We sort with respect to column numbers.
    Sort(nnz, IndCol, IndRow, Val);

    // A is filled.
    k = 0;
    for (int i = 0; i < m; i++)
      {
	int first_index = k;
	// We get the size of the column i.
	while (k < nnz && IndCol(k) <= i)
	  k++;

	int size_column = k - first_index;
	// If column not empty.
	if (size_column > 0)
	  {
	    A.ReallocateColumn(i, size_column);
	    k = first_index;
	    Sort(k, k+size_column-1, IndRow, Val);
	    for (int j = 0; j < size_column; j++)
	      {
		A.Index(i,j) = IndRow(k);
		A.Value(i,j) = Val(k);
		k++;
	      }
	  }
	else
	  A.ClearColumn(i);
      }
  }


  //! Permutation of a symmetric matrix stored by rows.
  /*!
    B(row_perm(i),col_perm(j)) = A(i,j) and A = B.
    Equivalent Matlab operation: A(row_perm, col_perm) = A.
  */
  template<class T, class Prop, class Allocator>
  void PermuteMatrix(Matrix<T, Prop,
		     ArrayRowSymComplexSparse, Allocator>& A,
		     const IVect& row_perm,const IVect& col_perm)
  {
    // It is assumed that the permuted matrix is still symmetric! For example,
    // the user can provide row_perm = col_perm.
    int m = A.GetM();
    int nnz_real = A.GetRealDataSize(), nnz_imag = A.GetImagDataSize();
    IVect IndRow(nnz_real), IndCol(nnz_real);
    Vector<T, VectFull, Allocator> Val(nnz_real);

    // First we convert the matrix in coordinate format and we permute the
    // indices.
    // IndRow -> indices of the permuted rows
    // IndCol -> indices of the permuted columns
    int k = 0;
    for (int i = 0; i < m; i++)
      {
	for (int j = 0; j < A.GetRealRowSize(i); j++)
	  {
	    IndRow(k) = row_perm(i);
	    Val(k) = A.ValueReal(i,j);
	    IndCol(k) = col_perm(A.IndexReal(i, j));
	    if (IndCol(k) <= IndRow(k))
	      {
		// We store only the superior part of the symmetric matrix.
		int ind_tmp = IndRow(k);
		IndRow(k) = IndCol(k);
		IndCol(k) = ind_tmp;
	      }
	    k++;
	  }
      }

    // We sort by row number.
    Sort(nnz_real, IndRow, IndCol, Val);

    // A is filled.
    k = 0;
    for (int i = 0; i < m; i++)
      {
	int first_index = k;
	// We get the size of the row i.
	while (k < nnz_real && IndRow(k) <= i)
	  k++;

	int size_row = k - first_index;
	// If row not empty.
	if (size_row > 0)
	  {
	    A.ReallocateRealRow(i, size_row);
	    k = first_index;
	    Sort(k, k+size_row-1, IndCol, Val);
	    for (int j = 0; j < size_row; j++)
	      {
		A.IndexReal(i,j) = IndCol(k);
		A.ValueReal(i,j) = Val(k);
		k++;
	      }
	  }
	else
	  A.ClearRealRow(i);
      }

    // Same procedure for imaginary part.

    IndRow.Reallocate(nnz_imag);
    IndCol.Reallocate(nnz_imag);
    Val.Reallocate(nnz_imag);

    // First we convert the matrix in coordinate format and we permute the
    // indices.
    // IndRow -> indices of the permuted rows
    // IndCol -> indices of the permuted columns
    k = 0;
    for (int i = 0; i < m; i++)
      {
	for (int j = 0; j < A.GetImagRowSize(i); j++)
	  {
	    IndRow(k) = row_perm(i);
	    Val(k) = A.ValueImag(i,j);
	    IndCol(k) = col_perm(A.IndexImag(i,j));
	    if (IndCol(k) <= IndRow(k))
	      {
		// We store only the superior part of the symmetric matrix.
		int ind_tmp = IndRow(k);
		IndRow(k) = IndCol(k);
		IndCol(k) = ind_tmp;
	      }
	    k++;
	  }
      }
    // We sort by row number.
    Sort(nnz_imag, IndRow, IndCol, Val);

    // A is filled
    k = 0;
    for (int i = 0; i < m; i++)
      {
	int first_index = k;
	// We get the size of the row i.
	while (k < nnz_imag && IndRow(k) <= i)
	  k++;
	int size_row = k - first_index;
	// If row not empty.
	if (size_row > 0)
	  {
	    A.ReallocateImagRow(i, size_row);
	    k = first_index;
	    Sort(k, k+size_row-1, IndCol, Val);
	    for (int j = 0; j < size_row; j++)
	      {
		A.IndexImag(i,j) = IndCol(k);
		A.ValueImag(i,j) = Val(k);
		k++;
	      }
	  }
	else
	  A.ClearImagRow(i);
      }
  }


  //! Permutation of a symmetric matrix stored by rows.
  /*!
    B(row_perm(i),col_perm(j)) = A(i,j) and A = B.
    Equivalent Matlab operation: A(row_perm, col_perm) = A.
  */
  template<class T, class Prop, class Allocator>
  void PermuteMatrix(Matrix<T, Prop, ArrayRowSymSparse, Allocator>& A,
		     const IVect& row_perm, const IVect& col_perm)
  {
    // It is assumed that the permuted matrix is still symmetric! For example,
    // the user can provide row_perm = col_perm.
    int m = A.GetM();
    int nnz = A.GetDataSize();
    IVect IndRow(nnz), IndCol(nnz);
    Vector<T,VectFull,Allocator> Val(nnz);

    // First we convert the matrix in coordinate format and we permute the
    // indices.
    // IndRow -> indices of the permuted rows
    // IndCol -> indices of the permuted columns
    int k = 0;
    for (int i = 0; i < m; i++)
      {
	for (int j = 0; j < A.GetRowSize(i); j++)
	  {
	    IndRow(k) = row_perm(i);
	    Val(k) = A.Value(i, j);
	    IndCol(k) = col_perm(A.Index(i, j));
	    if (IndCol(k) <= IndRow(k))
	      {
		// We store only the superior part of the symmetric matrix.
		int ind_tmp = IndRow(k);
		IndRow(k) = IndCol(k);
		IndCol(k) = ind_tmp;
	      }
	    k++;
	  }
      }
    // We sort with respect to row numbers.
    Sort(nnz, IndRow, IndCol, Val);

    // A is filled.
    k = 0;
    for (int i = 0; i < m; i++)
      {
	int first_index = k;
	// We get the size of the row i.
	while (k < nnz && IndRow(k) <= i)
	  k++;
	int size_row = k - first_index;
	// If row not empty.
	if (size_row > 0)
	  {
	    A.ReallocateRow(i, size_row);
	    k = first_index;
	    Sort(k, k+size_row-1, IndCol, Val);
	    for (int j = 0; j < size_row; j++)
	      {
		A.Index(i,j) = IndCol(k);
		A.Value(i,j) = Val(k);
		k++;
	      }
	  }
	else
	  A.ClearRow(i);
      }
  }


  //! Each row and column are scaled.
  /*!
    We compute diag(scale_left)*A*diag(scale_right).
  */
  template<class Prop, class T1, class Allocator1,
	   class T2, class Allocator2, class T3, class Allocator3>
  void ScaleMatrix(Matrix<T1, Prop, ArrayRowSparse, Allocator1>& A,
		   const Vector<T2, VectFull, Allocator2>& scale_left,
		   const Vector<T3, VectFull, Allocator3>& scale_right)
  {
    int m = A.GetM();
    for (int i = 0; i < m; i++ )
      for (int j = 0; j < A.GetRowSize(i); j++ )
	A.Value(i,j) *= scale_left(i) * scale_right(A.Index(i, j));

  }


  //! Each row and column are scaled.
  /*!
    We compute diag(scale_left)*A*diag(scale_right).
  */
  template<class Prop, class T1, class Allocator1,
	   class T2, class Allocator2, class T3, class Allocator3>
  void ScaleMatrix(Matrix<T1, Prop, ArrayRowSymSparse, Allocator1>& A,
		   const Vector<T2, VectFull, Allocator2>& scale_left,
		   const Vector<T3, VectFull, Allocator3>& scale_right)
  {
    int m = A.GetM();
    for (int i = 0; i < m; i++ )
      for (int j = 0; j < A.GetRowSize(i); j++ )
	A.Value(i,j) *= scale_left(i) * scale_right(A.Index(i, j));

  }


  //! Each row and column are scaled.
  /*!
    We compute diag(scale_left)*A*diag(scale_right).
  */
  template<class Prop, class T1, class Allocator1,
	   class T2, class Allocator2, class T3, class Allocator3>
  void ScaleMatrix(Matrix<T1, Prop, ArrayRowSymComplexSparse, Allocator1>& A,
		   const Vector<T2, VectFull, Allocator2>& scale_left,
		   const Vector<T3, VectFull, Allocator3>& scale_right)
  {
    int m = A.GetM();
    for (int i = 0; i < m; i++ )
      {
	for (int j = 0; j < A.GetRealRowSize(i); j++ )
	  A.ValueReal(i,j) *= scale_left(i) * scale_right(A.IndexReal(i, j));

	for (int j = 0; j < A.GetImagRowSize(i); j++ )
	  A.ValueImag(i,j) *= scale_left(i) * scale_right(A.IndexImag(i, j));
      }
  }


  //! Each row and column are scaled.
  /*!
    We compute diag(scale_left)*A*diag(scale_right).
  */
  template<class Prop, class T1, class Allocator1,
	   class T2, class Allocator2, class T3, class Allocator3>
  void ScaleMatrix(Matrix<T1, Prop, ArrayRowComplexSparse, Allocator1>& A,
		   const Vector<T2, VectFull, Allocator2>& scale_left,
		   const Vector<T3, VectFull, Allocator3>& scale_right)
  {
    int m = A.GetM();
    for (int i = 0; i < m; i++ )
      {
	for (int j = 0; j < A.GetRealRowSize(i); j++ )
	  A.ValueReal(i,j) *= scale_left(i) * scale_right(A.IndexReal(i, j));

	for (int j = 0; j < A.GetImagRowSize(i); j++ )
	  A.ValueImag(i,j) *= scale_left(i) * scale_right(A.IndexImag(i, j));
      }
  }


  //! Each row is scaled.
  /*!
    We compute diag(S)*A where S = scale.
  */
  template<class T1, class Allocator1,
	   class Prop, class T2, class Allocator2>
  void ScaleLeftMatrix(Matrix<T1, Prop, ArrayRowSparse, Allocator1>& A,
		       const Vector<T2, VectFull, Allocator2>& scale)
  {
    int m = A.GetM();
    for (int i = 0; i < m; i++ )
      for (int j = 0; j < A.GetRowSize(i); j++ )
	A.Value(i,j) *= scale(i);
  }


  //! Each row is scaled.
  /*!
    We compute diag(S)*A where S = scale.  In order to keep symmetry, the
    operation is performed on upper part of the matrix, considering that lower
    part is affected by this operation.
  */
  template<class T1, class Allocator1,
	   class Prop, class T2, class Allocator2>
  void ScaleLeftMatrix(Matrix<T1, Prop, ArrayRowSymSparse, Allocator1>& A,
		       const Vector<T2, VectFull, Allocator2>& scale)
  {
    int m = A.GetM();
    for (int i = 0; i < m; i++ )
      for (int j = 0; j < A.GetRowSize(i); j++ )
	A.Value(i,j) *= scale(i);
  }


  //! Each row is scaled.
  /*!
    We compute diag(S)*A where S = scale.  In order to keep symmetry, the
    operation is performed on upper part of the matrix, considering that lower
    part is affected by this operation.
  */
  template<class T1, class Allocator1,
	   class Prop, class T2, class Allocator2>
  void ScaleLeftMatrix(Matrix<T1, Prop, ArrayRowSymComplexSparse, Allocator1>& A,
		       const Vector<T2, VectFull, Allocator2>& scale)
  {
    int m = A.GetM();
    for (int i = 0; i < m; i++ )
      {
	for (int j = 0; j < A.GetRealRowSize(i); j++ )
	  A.ValueReal(i,j) *= scale(i);

	for (int j = 0; j < A.GetImagRowSize(i); j++ )
	  A.ValueImag(i,j) *= scale(i);
      }
  }


  //! Each row is scaled.
  /*!
    We compute diag(S)*A where S = scale.
  */
  template<class T1, class Allocator1,
	   class Prop, class T2, class Allocator2>
  void ScaleLeftMatrix(Matrix<T1, Prop, ArrayRowComplexSparse, Allocator1>& A,
		       const Vector<T2, VectFull, Allocator2>& scale)
  {
    int m = A.GetM();
    for (int i = 0; i < m; i++ )
      {
	for (int j = 0; j < A.GetRealRowSize(i); j++ )
	  A.ValueReal(i,j) *= scale(i);

	for (int j = 0; j < A.GetImagRowSize(i); j++ )
	  A.ValueImag(i,j) *= scale(i);
      }

  }


} // end namespace

#define SELDON_FILE_PERMUTATION_SCALING_MATRIX_CXX
#endif
