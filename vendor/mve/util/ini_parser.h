/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef UTIL_INI_PARSER_HEADER
#define UTIL_INI_PARSER_HEADER

#include <istream>
#include <string>
#include <map>

#include "util/defines.h"

UTIL_NAMESPACE_BEGIN

/** Parses a file in INI format and places key/value pairs in the map. */
void
parse_ini (std::istream& stream, std::map<std::string, std::string>* map);

/**
 * Writes an INI file for the key/value pairs in the map.
 * Section names are part of the key, separated with a dot from the key.
 */
void
write_ini (std::map<std::string, std::string> const& map, std::ostream& stream);

UTIL_NAMESPACE_END

#endif  /* UTIL_INI_PARSER_HEADER */

