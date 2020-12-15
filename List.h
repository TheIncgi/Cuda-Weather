template <class T>
class List {

  private:
  T* values;
  int _size;
  int limit;
  __device__ void resize(int newLimit){
    T* n = new T[newLimit];
    for(int i = 0; i<_size; i++)
      n[i] = values[i];
    delete values;
    limit = newLimit;
    values = n;
  }


  public:
  __device__ List(int limit) {
      this->limit = limit;
      _size = 0;
    values = new T[limit];
    }
  __device__ ~List() {delete values;}
  __device__ void push(T value){
      if(_size==limit) resize(_size+16);
      values[_size++] = value;
    }
  __device__ T pop(){ return values[--_size];}
  __device__ void insert(int index, T value){
      if(_size==limit) resize(_size+16);
      for(int i = _size; i>index; i--){
        values[i] = values[i-1];
      }
      values[index] = value;
      _size++;
    }
  __device__ T operator[] (int x){return values[x];}
  __device__ int size(){ return _size;}
  __device__ bool contains(T value) {
      for(int i = 0; i<_size; i++)
        if(values[i] == value) return true;
      return false;
    }

};
