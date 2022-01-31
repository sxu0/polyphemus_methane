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


#ifndef SELDON_FILE_SPARSE_VECTOR_HXX

namespace Seldon
{


  //! Sparse vector class.
  template <class T, class Allocator>
  class Vector<T, VectSparse, Allocator>:
    public Vector<T, VectFull, Allocator>
  {
    // typedef declarations.
  public:
    typedef typename Allocator::value_type value_type;
    typedef typename Allocator::pointer pointer;
    typedef typename Allocator::const_pointer const_pointer;
    typedef typename Allocator::reference reference;
    typedef typename Allocator::const_reference const_reference;

  protected :
    static SELDON_DEFAULT_ALLOCATOR<int> index_allocator_;

    // Attributes.
  private:
    //! Indices of the non-zero entries.
    int* index_;

    // Methods.
  public:
    // Constructor.
    explicit Vector()  throw();
    explicit Vector(int i);
    Vector(const Vector<T, VectSparse, Allocator>& A);

    // Destructor.
    ~Vector();
    void Clear();

    // Memory management.
    void Reallocate(int i);
    void Resize(int i);
    void SetData(int nz, T* data, int* index);
    template<class Allocator2>
    void SetData(Vector<T, VectFull, Allocator2>& data,
		 Vector<int>& index);
    void Nullify();

    // Element access and affectation.
    reference Value(int i);
    const_reference Value(int i) const;
    int& Index(int i);
    int Index(int i) const;
    reference operator() (int i);
#ifndef SWIG
    value_type operator() (int i) const;
    Vector<T, VectSparse, Allocator>& operator= (const Vector<T, VectSparse,
						 Allocator>& X);
#endif
    void Copy(const Vector<T, VectSparse, Allocator>& X);

    // Basic functions.
    int* GetIndex() const;

    // Convenient functions.
    template <class T0>
#ifndef SWIG
    Vector<T, VectSparse, Allocator>& operator= (const T0& X);
#endif
    void Print() const;
    void Assemble();
    template<class T0>
    void RemoveSmallEntry(const T0& epsilon);
    void AddInteraction(int i, const T& val);
    void AddInteractionRow(int, int*, T*, bool);
    template<class Allocator0>
    void AddInteractionRow(int nb, Vector<int> col,
			   Vector<T, VectFull, Allocator0> val, bool);

    // Input/output functions.
    void Write(string FileName) const;
#ifndef SWIG
    void Write(ostream& FileStream) const;
#endif
    void WriteText(string FileName) const;
#ifndef SWIG
    void WriteText(ostream& FileStream) const;
#endif
    void Read(string FileName);
#ifndef SWIG
    void Read(istream& FileStream);
#endif
    void ReadText(string FileName);
#ifndef SWIG
    void ReadText(istream& FileStream);
#endif

  };

#ifndef SWIG
  template <class T, class Allocator>
  ostream& operator << (ostream& out,
			const Vector<T, VectSparse, Allocator>& V);
#endif


  template<class T, class Allocator>
  SELDON_DEFAULT_ALLOCATOR<int>
  Vector<T, VectSparse, Allocator>::index_allocator_;


} // namespace Seldon.

#define SELDON_FILE_SPARSE_VECTOR_HXX
#endif
