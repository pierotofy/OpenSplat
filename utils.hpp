#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <algorithm>
#include <random>
#include <iostream>

template <typename T>
class InfiniteRandomIterator
{
    using VecType = std::vector<T>;
public:

    InfiniteRandomIterator(const VecType &v) : v(v), engine(42) {
        shuffleV();
    }

    void shuffleV(){
        std::shuffle(std::begin(v), std::end(v), engine);
        i = 0;
    }

    T next(){
        T ret = v[i++];
        if (i >= v.size()) shuffleV();
        return ret;
    }
private:
    VecType v;
    size_t i;
    std::default_random_engine engine;
};

#endif