#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <algorithm>
#include <random>
#include <queue>
#include <iostream>
#include <utility>


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

template <typename T>
class PriorityQueue
{
public:
    PriorityQueue() {
    }

    T pop(){
        T v = q.top().second;
        q.pop();
        return v;
    }

    void push(T v, float priority){
        q.push(std::make_pair(priority, v));
    }

    size_t size(){
        return q.size();
    }
private:
    std::priority_queue<std::pair<float, T>> q;
};

#endif