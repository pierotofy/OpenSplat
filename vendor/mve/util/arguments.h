/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef MVE_ARGUMENTS_HEADER
#define MVE_ARGUMENTS_HEADER

#include <ostream>
#include <vector>
#include <string>
#include <stdexcept>

#include "util/strings.h"
#include "util/defines.h"
#include "util/exception.h"

UTIL_NAMESPACE_BEGIN

/**
 * A single argument option.
 */
struct ArgOption
{
    char sopt; ///< Short option name
    std::string lopt; ///< Long option name
    std::string desc; ///< Description
    bool argument; ///< Requires argument?
};

/**
 * An argument which can either be an option or a non-option.
 */
struct ArgResult
{
    ArgOption const* opt; ///< Null for non-options
    std::string arg; ///< Empty for options without arguments

    /** Returns argument converted to type T. */
    template <typename T>
    T get_arg (void) const;
};

/**
 * Argument class that provides a parser and convenient access for
 * command line arguments as used by GNU utils. An error while parsing
 * the arguments raises exception util::Exception. The resulting list
 * of arguments is provided as iterator over ArgResult objects.
 *
 * The parser enforces arguments in the format:
 *
 *    command [options | non-options] [--] [non-options]
 *
 * The following rules apply:
 *
 * - Arguments are short options if they begin with a hyphen delimiter "-"
 * - Arguments are long options if they begin with double hyphen "--"
 * - Multiple short options may follow a hyphen delimiter if the options
 *   do not take arguments. Thus, "-abc" and "-a -b -c" are equivalent.
 * - Options may require an argument. For a short option, the argument
 *   may or may not appear as separate tokens, thus "-o foo" and "-ofoo" are
 *   equivalent. Arguments to long options are specified with "--NAME=VALUE".
 * - The token "--" terminates all options; any following arguments
 *   are treated as non-option arguments.
 * - A token consisting of a single hyphen character is interpreted as
 *   an ordinary non-option argument.
 *
 * The first argument, argv[0], is treated as the command that started
 * the application and is ignored; thus argument parsing starts at argv[1].
 */
class Arguments
{
public:
    Arguments (void);

    /** Sets the usage string for printing the help text. */
    void set_usage (std::string const& str);

    /** Sets the usage string for printing the help text. */
    void set_usage (char const* argv0, std::string const& usage);

    /** Sets the optional description text to be printed on help. */
    void set_description (std::string const& str);

    /**
     * Sets the text width of the description string for automatic
     * line breaking. This defaults to 75, 0 disables feature.
     */
    void set_description_word_wrap (int width);

    /**
     * Sets amount column width for indenting help description.
     * This property defaults to 16.
     */
    void set_helptext_indent (int indent);

    /**
     * Sets a maximum limit on the amount of non-option argument.
     * This propertiy default to no limit.
     */
    void set_nonopt_maxnum (std::size_t limit);

    /**
     * Sets a minimum limit on the amount of non-option argument.
     * This property defaults to zero.
     */
    void set_nonopt_minnum (std::size_t limit);

    /**
     * Specifies if the application should be exited if parsing
     * was not successful. Before exiting, the arguments help
     * text is printed to stderr. Defaults to "not exit".
     */
    void set_exit_on_error (bool exit);

    /** Adds an option to the list of possible arguments. */
    void add_option (char shortname, std::string const& longname,
        bool has_argument, std::string const& description = "");

    /** Parses arguments from a vector. */
    void parse (std::vector<std::string> const& args);
    /** Parses command line arguments. */
    void parse (int argc, char const* const* argv);

    /**
     * Iterator for the results. If opt is null, the argument is a
     * non-option. Otherwise opt points to one of the given argument
     * options. The method returns null if there are no more arguments.
     */
    ArgResult const* next_result (void);

    /**
     * Iterator for the options. It does not return arguments that
     * are non-options, i.e. opt is never null. The method returns
     * null if there are no more non-option arguments.
     */
    ArgResult const* next_option (void);

    /** Returns the Nth non-option argument, or an empty string. */
    std::string get_nth_nonopt (std::size_t index);

    /** Returns the Nth non-option argument as type T, or throws. */
    template <typename T>
    T get_nth_nonopt_as (std::size_t index);

    /** Returns a numeric list of IDs by parsing the given string. */
    void get_ids_from_string (std::string const& str, std::vector<int>* ids);

    /** Generates a help text that lists all arguments. */
    void generate_helptext (std::ostream& stream) const;

private:
    void parse_intern (std::vector<std::string> const& args);
    void parse_long_opt (std::string const& tok);
    bool parse_short_opt (std::string const& tok1, std::string const& tok2);
    ArgOption const* find_opt (char sopt);
    ArgOption const* find_opt (std::string const& lopt);

private:
    /* Settings. */
    std::size_t nonopt_min;
    std::size_t nonopt_max;
    bool auto_exit;
    std::vector<ArgOption> options;
    std::string usage_str;
    std::string descr_str;
    int helptext_indent;
    int descrtext_width;

    /* Parse result. */
    std::vector<ArgResult> results;
    std::string command_name;

    /* Iterator. */
    std::size_t cur_result;
};

/* ---------------------------------------------------------------- */

template <typename T>
inline T
ArgResult::get_arg (void) const
{
    return string::convert<T>(this->arg);
}

template <>
inline std::string
ArgResult::get_arg (void) const
{
    return this->arg;
}

inline void
Arguments::set_usage (std::string const& str)
{
    this->usage_str = str;
}

inline void
Arguments::set_description (std::string const& str)
{
    this->descr_str = str;
}

inline void
Arguments::set_description_word_wrap (int width)
{
    this->descrtext_width = width;
}

inline void
Arguments::set_helptext_indent (int indent)
{
    this->helptext_indent = indent;
}

inline void
Arguments::set_nonopt_maxnum (std::size_t limit)
{
    this->nonopt_max = limit;
}

inline void
Arguments::set_nonopt_minnum (std::size_t limit)
{
    this->nonopt_min = limit;
}

inline void
Arguments::set_exit_on_error (bool exit)
{
    this->auto_exit = exit;
}

template <typename T>
T
Arguments::get_nth_nonopt_as (std::size_t index)
{
    std::string str(this->get_nth_nonopt(index));
    if (str.empty())
        throw std::runtime_error("No such arguemnt");
    return util::string::convert<T>(str);
}

UTIL_NAMESPACE_END

#endif /* MVE_ARGUMENTS_HEADER */
