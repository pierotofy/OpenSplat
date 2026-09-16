#ifndef ZIP_UTILS_H
#define ZIP_UTILS_H

#include <string>

bool isZipArchive(const std::string &path);
// Extracts the archive into cacheDir (or reuses a previous extraction)
// and returns the project root inside it
std::string extractZipToCache(const std::string &zipPath, const std::string &cacheDir);

#endif
