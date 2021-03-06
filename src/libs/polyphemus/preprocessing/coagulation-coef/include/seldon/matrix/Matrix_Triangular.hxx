// Copyright (C) 2001-2009 Vivien Mallet
// Copyright (C) 2003-2009 Marc Duruflé
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


// To be included by Seldon.hxx

#ifndef SELDON_FILE_MATRIX_TRIANGULAR_HXX

#include "../share/Common.hxx"
#include "../share/Properties.hxx"
#include "../share/Storage.hxx"
#include "../share/Errors.hxx"
#include "../share/Allocator.hxx"

namespace Seldon
{


  //! Triangular matrix stored in a full matrix.
  template <class T, class Prop, class Storage,
	    class Allocator = SELDON_DEFAULT_ALLOCATOR<T> >
  class Matrix_Triangular: public Matrix_Base<T, Allocator>
  {
    // typedef declaration.
  public:
    typedef typename Allocator::value_type value_type;
    typedef typename Allocator::pointer pointer;
    typedef typename Allocator::const_pointer const_pointer;
    typedef typename Allocator::reference reference;
    typedef typename Allocator::const_reference const_reference;

    // Attributes.
  protected:
    pointer* me_;

    // Methods.
  public:
    // Constructor.
    Matrix_Triangular();
    Matrix_Triangular(int i, int j);
    Matrix_Triangular(const Matrix_Triangular<T, Prop,
		      Storage, Allocator>& A);

    // Destructor.
    ~Matrix_Triangular();
    void Clear();

    // Basic methods.
    int GetDataSize() const;

    // Memory management.
    void Reallocate(int i, int j);
    void Resize(int i, int j);
    void SetData(int i, int j, pointer data);
    void Nullify();

    // Element access and affectation.
    value_type operator() (int i, int j);
    value_type operator() (int i, int j) const;
    const_reference Val(int i, int j) const;
    reference Val(int i, int j);
    reference operator[] (int i);
    const_reference operator[] (int i) const;
    Matrix_Triangular<T, Prop, Storage, Allocator>&
    operator= (const Matrix_Triangular<T, Prop, Storage, Allocator>& A);
    void Copy(const Matrix_Triangular<T, Prop, Storage, Allocator>& A);

    // Convenient functions.
    void Zero();
    void SetIdentity();
    void Fill();
    template <class T0>
    void Fill(const T0& x);
    template <class T0>
    Matrix_Triangular<T, Prop, Storage, Allocator>& operator= (const T0& x);
    void FillRand();
    void Print() const;
    void Print(int a, int b, int m, int n) const;
    void Print(int l) const;

    // Input/output functions.
    void Write(string FileName) const;
    void Write(ostream& FileStream) const;
    void WriteText(string FileName) const;
    void WriteText(ostream& FileStream) const;
    void Read(string FileName);
    void Read(istream& FileStream);
    void ReadText(string FileName);
    void ReadText(istream& FileStream);

  };


  //! Column-major upper-triangular full-matrix class.
  template <class T, class Prop, class Allocator>
  class Matrix<T, Prop, ColUpTriang, Allocator>:
    public Matrix_Triangular<T, Prop, ColUpTriang, Allocator>
  {
  public:
    Matrix()  throw();
    Matrix(int i, int j);

    template <class T0>
    Matrix<T, Prop, ColUpTriang, Allocator>& operator= (const T0& x);
    template<class T0>
    Matrix<T, Prop, ColUpTriang, Allocator>& operator*= (const T0& x);

  };


  //! Column-major lower-triangular full-matrix class.
  template <class T, class Prop, class Allocator>
  class Matrix<T, Prop, ColLoTriang, Allocator>:
    public Matrix_Triangular<T, Prop, ColLoTriang, Allocator>
  {
  public:
    Matrix()  throw();
    Matrix(int i, int j);

    template <class T0>
    Matrix<T, Prop, ColLoTriang, Allocator>& operator= (const T0& x);
    template<class T0>
    Matrix<T, Prop, ColLoTriang, Allocator>& operator*= (const T0& x);

  };


  //! Row-major upper-triangular full-matrix class.
  template <class T, class Prop, class Allocator>
  class Matrix<T, Prop, RowUpTriang, Allocator>:
    public Matrix_Triangular<T, Prop, RowUpTriang, Allocator>
  {
  public:
    Matrix()  throw();
    Matrix(int i, int j);

    template <class T0>
    Matrix<T, Prop, RowUpTriang, Allocator>& operator= (const T0& x);
    template<class T0>
    Matrix<T, Prop, RowUpTriang, Allocator>& operator*= (const T0& x);

  };


  //! Row-major lower-triangular full-matrix class.
  template <class T, class Prop, class Allocator>
  class Matrix<T, Prop, RowLoTriang, Allocator>:
    public Matrix_Triangular<T, Prop, RowLoTriang, Allocator>
  {
  public:
    Matrix()  throw();
    Matrix(int i, int j);

    template <class T0>
    Matrix<T, Prop, RowLoTriang, Allocator>& operator= (const T0& x);
    template<class T0>
    Matrix<T, Prop, RowLoTriang, Allocator>& operator*= (const T0& x);
  };


} // namespace Seldon.

#define SELDON_FILE_MATRIX_TRIANGULAR_HXX
#endif
