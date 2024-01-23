/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <iostream>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <limits>

#include "util/strings.h"
#include "util/tokenizer.h"
#include "util/arguments.h"

UTIL_NAMESPACE_BEGIN

Arguments::Arguments (void)
    : nonopt_min(0)
    , nonopt_max(std::numeric_limits<std::size_t>::max())
    , auto_exit(false)
    , helptext_indent(16)
    , descrtext_width(75)
    , cur_result(std::numeric_limits<std::size_t>::max())
{
}

/* ---------------------------------------------------------------- */

void
Arguments::set_usage (char const* argv0, std::string const& usage)
{
    std::stringstream ss;
    ss << "Usage: " << argv0 << " " << usage;
    this->usage_str = ss.str();
}

/* ---------------------------------------------------------------- */

void
Arguments::add_option (char shortname, std::string const& longname,
        bool has_argument, std::string const& description)
{
    if (shortname == '\0' && longname.empty())
        throw std::invalid_argument("Neither short nor long name given");

    /* Check if option already exists. */
    for (std::size_t i = 0; i < this->options.size(); ++i)
        if ((shortname != '\0' && this->options[i].sopt == shortname)
            || (!longname.empty() && this->options[i].lopt == longname))
            throw std::runtime_error("Option already exists");

    ArgOption opt;
    opt.sopt = shortname;
    opt.lopt = longname;
    opt.argument = has_argument;
    opt.desc = description;
    this->options.push_back(opt);
}

/* ---------------------------------------------------------------- */

void
Arguments::parse (int argc, char const* const* argv)
{
    std::vector<std::string> args;
    for (int i = 0; i < argc; ++i)
        args.push_back(argv[i]);
    this->parse(args);
}

/* ---------------------------------------------------------------- */

void
Arguments::parse (std::vector<std::string> const& args)
{
    try
    {
        this->parse_intern(args);
    }
    catch (util::Exception const& e)
    {
        if (!this->auto_exit)
            throw;

        std::cerr << std::endl;
        this->generate_helptext(std::cerr);
        std::cerr << std::endl << "Error: " << e.what() << std::endl;
        std::exit(1);
    }
}

/* ---------------------------------------------------------------- */

void
Arguments::parse_long_opt (std::string const& tok)
{
    std::size_t pos = tok.find_first_of('=');
    std::string opt = tok.substr(2, pos - 2);
    std::string arg;
    if (pos != std::string::npos)
        arg = tok.substr(pos + 1);

    if (opt.empty())
        throw util::Exception("Invalid option: ", tok);

    ArgOption const* option = this->find_opt(opt);
    if (option == nullptr)
        throw util::Exception("Invalid option: ", tok);

    if (option->argument && arg.empty())
        throw util::Exception("Option missing argument: ", tok);

    if (!option->argument && !arg.empty())
        throw util::Exception("Option with unexpected argument: ", tok);

    ArgResult result;
    result.arg = arg;
    result.opt = option;
    this->results.push_back(result);
}

/* ---------------------------------------------------------------- */

bool
Arguments::parse_short_opt (std::string const& tok1, std::string const& tok2)
{
    if (tok1.size() < 2)
        throw std::runtime_error("Short option with too few chars");

    char opt = tok1[1];
    std::string arg;
    bool retval = false;
    ArgOption const* option = this->find_opt(opt);

    if (option == nullptr)
        throw util::Exception("Invalid option: ", tok1);

    /* If multiple options are given, recurse. */
    if (!option->argument && tok1.size() > 2)
    {
        for (std::size_t i = 1; i < tok1.size(); ++i)
        {
            std::string sopt = "-";
            sopt.append(1, tok1[i]);
            this->parse_short_opt(sopt, "");
        }
        return false;
    }

    if (option->argument && tok1.size() == 2)
    {
        if (tok2.empty() || tok2[0] == '-')
            throw util::Exception("Option missing argument: ", tok1);
        arg = tok2;
        retval = true;
    }
    else if (option->argument)
    {
        arg = tok1.substr(2);
    }

    if (!option->argument && tok1.size() != 2)
        throw util::Exception("Option with unexpected argument: ", tok1);

    ArgResult result;
    result.arg = arg;
    result.opt = option;
    this->results.push_back(result);

    return retval;
}

/* ---------------------------------------------------------------- */

ArgOption const*
Arguments::find_opt (char sopt)
{
    for (std::size_t i = 0; i < this->options.size(); ++i)
        if (this->options[i].sopt == sopt)
            return &this->options[i];

    return nullptr;
}

/* ---------------------------------------------------------------- */

ArgOption const*
Arguments::find_opt (std::string const& lopt)
{
    for (std::size_t i = 0; i < this->options.size(); ++i)
        if (this->options[i].lopt == lopt)
            return &this->options[i];

    return nullptr;
}

/* ---------------------------------------------------------------- */

void
Arguments::parse_intern (std::vector<std::string> const& args)
{
    this->results.clear();
    this->command_name.clear();
    this->cur_result = std::numeric_limits<std::size_t>::max();

    /* Iterate over all tokens and buils result structure. */
    bool parseopt = true;
    for (std::size_t i = 0; i < args.size(); ++i)
    {
        if (i == 0)
        {
            this->command_name = args[i];
            continue;
        }

        std::string tok = args[i];
        string::clip_newlines(&tok);
        string::clip_whitespaces(&tok);

        /* Skip empty tokens. */
        if (tok.empty())
            continue;

        if (parseopt && tok.size() == 2 && tok == "--")
        {
            /* Option terminator. */
            parseopt = false;
            continue;
        }
        else if (parseopt && tok.size() >= 3 && tok[0] == '-' && tok[1] == '-')
        {
            /* Long option. Requires only one token. */
            this->parse_long_opt(tok);
        }
        else if (parseopt && tok.size() >= 2 && tok[0] == '-')
        {
            /* Short option, possibly requiring the next token. */
            std::string next_tok;
            if (i + 1 < args.size())
            {
                next_tok = args[i + 1];
                string::clip_newlines(&next_tok);
                string::clip_whitespaces(&next_tok);
            }
            bool used_next = this->parse_short_opt(tok, next_tok);
            if (used_next)
                i += 1;
        }
        else
        {
            /* Regular non-option. */
            ArgResult res;
            res.opt = nullptr;
            res.arg = tok;
            this->results.push_back(res);
        }
    }

    /* Check non-option limit. */
    std::size_t num_nonopt = 0;
    for (std::size_t i = 0; i < this->results.size(); ++i)
    {
        if (this->results[i].opt == nullptr)
            num_nonopt += 1;
    }

    if (num_nonopt > this->nonopt_max)
        throw util::Exception("Too many non-option arguments");
    if (num_nonopt < this->nonopt_min)
        throw util::Exception("Too few non-option arguments");
}

/* ---------------------------------------------------------------- */

ArgResult const*
Arguments::next_result (void)
{
    this->cur_result += 1;
    if (this->cur_result >= this->results.size())
    {
        this->cur_result = (std::size_t)(-1);
        return nullptr;
    }

    return &this->results[this->cur_result];
}

/* ---------------------------------------------------------------- */

ArgResult const*
Arguments::next_option (void)
{
    while (true)
    {
        this->cur_result += 1;
        if (this->cur_result >= this->results.size())
            break;
        ArgResult const* ret = &this->results[this->cur_result];
        if (ret->opt == nullptr)
            continue;
        return ret;
    }
    this->cur_result = (std::size_t)(-1); // Reset iterator
    return nullptr;
}

/* ---------------------------------------------------------------- */

std::string
Arguments::get_nth_nonopt (std::size_t index)
{
    std::size_t cnt = 0;
    for (std::size_t i = 0; i < this->results.size(); ++i)
    {
        if (this->results[i].opt)
            continue;
        if (index == cnt)
            return this->results[i].arg;
        cnt += 1;
    }

    return std::string();
}

/* ---------------------------------------------------------------- */

void
Arguments::get_ids_from_string (std::string const& str, std::vector<int>* ids)
{
    ids->clear();
    if (str.empty() || util::string::lowercase(str) == "all")
        return;

    util::Tokenizer t;
    t.split(str, ',');
    for (std::size_t i = 0; i < t.size(); ++i)
    {
        std::size_t pos = t[i].find_first_of('-');
        if (pos != std::string::npos)
        {
            int const first = util::string::convert<int>(t[i].substr(0, pos));
            int const last = util::string::convert<int>(t[i].substr(pos + 1));
            int const inc = (first > last ? -1 : 1);
            for (int j = first; j != last + inc; j += inc)
                ids->push_back(j);
        }
        else
        {
            ids->push_back(util::string::convert<int>(t[i]));
        }
    }
}

/* ---------------------------------------------------------------- */

void
Arguments::generate_helptext (std::ostream& stream) const
{
    std::string descr = util::string::wordwrap
        (this->descr_str.c_str(), this->descrtext_width);

    if (!descr.empty())
        stream << descr << std::endl << std::endl;

    if (!this->usage_str.empty())
        stream << this->usage_str << std::endl;

    if (!this->options.empty() && (!this->usage_str.empty() || !descr.empty()))
        stream << "Available options: " << std::endl;

    for (std::size_t i = 0; i < this->options.size(); ++i)
    {
        ArgOption const& opt(this->options[i]);
        std::stringstream optstr;
        if (opt.sopt != '\0')
        {
            optstr << '-' << opt.sopt;
            if (opt.argument && opt.lopt.empty())
                optstr << " ARG";
            if (!opt.lopt.empty())
                optstr << ", ";

        }
        if (!opt.lopt.empty())
        {
            optstr << "--" << opt.lopt;
            if (opt.argument)
                optstr << "=ARG";
        }
        optstr << "  ";

        stream << "  " << std::setiosflags(std::ios::left)
            << std::setw(this->helptext_indent) << optstr.str()
            << opt.desc << std::endl;
    }
}

UTIL_NAMESPACE_END
