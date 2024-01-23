/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef UTIL_TOKENIZE_HEADER
#define UTIL_TOKENIZE_HEADER

#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

#include "util/strings.h"
#include "util/defines.h"

UTIL_NAMESPACE_BEGIN

/**
 * Simple tokenizer. Reads an input string and splits the string
 * according to some rules. The individual tokens are then stored
 * in the vector baseclass.
 */
class Tokenizer : public std::vector<std::string>
{
public:
    /**
     * Very simple tokenziation at a given delimiter characater.
     * If requested, subsequent delimiter characters lead to empty tokens.
     */
    void split (std::string const& str, char delim = ' ',
        bool keep_empty = false);

    /**
     * A tokenizer that parses shell commands into tokens.
     * It handles quotes gracefully, placing quoted strings into
     * a single token and removes the quotes.
     */
    void parse_cmd (std::string const& str);

    /**
     * Concatenates 'num' tokens with a space character
     * starting from token at position 'pos'. Passing '0' as
     * 'num' argument means to concat all remaining tokens.
     */
    std::string concat (std::size_t pos, std::size_t num = 0) const;

    /**
     * Returns the requested token as the specified type.
     */
    template <typename T>
    T get_as (std::size_t pos) const;
};

/* ---------------------------------------------------------------- */

inline void
Tokenizer::split (std::string const& str, char delim, bool keep_empty)
{
    this->clear();
    std::size_t new_tok = 0;
    std::size_t cur_pos = 0;
    for (; cur_pos < str.size(); ++cur_pos)
        if (str[cur_pos] == delim)
        {
            std::string token = str.substr(new_tok, cur_pos - new_tok);
            if (keep_empty || !token.empty())
                this->push_back(token);
            new_tok = cur_pos + 1;
        }

    if (keep_empty || new_tok < str.size())
        this->push_back(str.substr(new_tok));
}

/* ---------------------------------------------------------------- */

inline void
Tokenizer::parse_cmd (std::string const& str)
{
    this->clear();
    bool in_quote = false;
    std::string token;
    for (std::size_t i = 0; i < str.size(); ++i)
    {
        char chr = str[i];

        if (chr == ' ' && !in_quote)
        {
            this->push_back(token);
            token.clear();
        }
        else if (chr == '"')
            in_quote = !in_quote;
        else
            token.push_back(chr);
    }
    this->push_back(token);
}

/* ---------------------------------------------------------------- */

inline std::string
Tokenizer::concat (std::size_t pos, std::size_t num) const
{
    std::stringstream ss;
    std::size_t max = (num == 0
        ? this->size()
        : std::min(pos + num, this->size()));

    for (std::size_t i = pos; i < max; ++i)
    {
        ss << (i == pos ? "" : " ");
        ss << this->at(i);
    }

    return ss.str();
}

/* ---------------------------------------------------------------- */

template <typename T>
inline T
Tokenizer::get_as (std::size_t pos) const
{
    return string::convert<T>(this->at(pos));
}

UTIL_NAMESPACE_END

#endif /* UTIL_TOKENIZE_HEADER */
