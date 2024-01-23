/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <sstream>

#include "util/exception.h"
#include "util/strings.h"
#include "util/ini_parser.h"

UTIL_NAMESPACE_BEGIN

namespace
{
    util::Exception
    get_exception (int line_number, char const* message)
    {
        std::stringstream ss;
        ss << "Line " << line_number << ": " << message;
        return ss.str();
    }
}

void
parse_ini (std::istream& stream, std::map<std::string, std::string>* map)
{
    std::string section_name;
    int line_number = 0;
    while (true)
    {
        /* Read line from input. */
        line_number += 1;
        std::string line;
        std::getline(stream, line);
        if (stream.eof())
            break;

        util::string::clip_newlines(&line);
        util::string::clip_whitespaces(&line);

        /* Skip empty lines and comments. */
        if (line.empty())
            continue;
        if (line[0] == '#')
            continue;

        /* Read section name. */
        if (line[0] == '[' && line[line.size() - 1] == ']')
        {
            section_name = line.substr(1, line.size() - 2);
            continue;
        }

        /* Read key/value pair. */
        std::size_t sep_pos = line.find_first_of('=');
        if (sep_pos == std::string::npos)
            throw get_exception(line_number, "Invalid line");

        std::string key = line.substr(0, sep_pos);
        util::string::clip_newlines(&key);
        util::string::clip_whitespaces(&key);

        std::string value = line.substr(sep_pos + 1);
        util::string::clip_newlines(&value);
        util::string::clip_whitespaces(&value);

        if (key.empty())
            throw get_exception(line_number, "Empty key");
        if (section_name.empty())
            throw get_exception(line_number, "No section");

        key = section_name + "." + key;
        (*map)[key] = value;
    }
}

void
write_ini (std::map<std::string, std::string> const& map, std::ostream& stream)
{
    std::string last_section;
    std::map<std::string, std::string>::const_iterator iter;
    for (iter = map.begin(); iter != map.end(); iter++)
    {
        std::string key = iter->first;
        std::string value = iter->second;
        util::string::clip_newlines(&key);
        util::string::clip_whitespaces(&key);
        util::string::clip_newlines(&value);
        util::string::clip_whitespaces(&value);

        std::size_t section_pos = key.find_first_of('.');
        if (section_pos == std::string::npos)
            throw std::runtime_error("Key/value pair without section");
        std::string section = key.substr(0, section_pos);
        key = key.substr(section_pos + 1);

        if (section != last_section)
        {
            stream << "\n[" << section << "]\n";
            last_section = section;
        }

        stream << key << " = " << value << std::endl;
    }
}

UTIL_NAMESPACE_END
