/*
 * Copyright (C) 2015, Simon Fuhrmann, Andre Schulz
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cctype>
#include <cerrno> // errno
#include <cstring> // std::strerror
#include <cstdio> // std::rename
#include <algorithm>

#if defined(_WIN32)
#   include <direct.h>
#   include <io.h>
#   include <shlobj.h>
#   include <sys/stat.h>
#   include <sys/types.h>
#else // Linux, OSX, ...
#   include <dirent.h>
#   include <unistd.h>
#   include <sys/stat.h>
#   include <sys/types.h>
#   include <pwd.h>
#endif

#if defined(__APPLE__)
#   include <mach-o/dyld.h> // for _NSGetExecutablePath
#endif

#ifndef PATH_MAX
#   define PATH_MAX 2048
#endif

#include "util/exception.h"
#include "util/system.h"
#include "util/file_system.h"

UTIL_NAMESPACE_BEGIN
UTIL_FS_NAMESPACE_BEGIN

// PATH_MAX might be long?
char home_path[PATH_MAX] = { 0 };
char app_data_path[PATH_MAX] = { 0 };

bool
exists (char const* pathname)
{
#ifdef _WIN32
    struct _stat statbuf;
    if (::_stat(pathname, &statbuf) < 0)
        return false;
#else // _WIN32
    struct stat statbuf;
    if (::stat(pathname, &statbuf) < 0)
        return false;
#endif // _WIN32

  return true;
}

/* ---------------------------------------------------------------- */

bool
dir_exists (char const* pathname)
{
#ifdef _WIN32
    struct _stat statbuf;
    if (::_stat(pathname, &statbuf) < 0)
        return false;

    if (!(statbuf.st_mode & _S_IFDIR))
        return false;
#else // _WIN32
    struct stat statbuf;
    if (::stat(pathname, &statbuf) < 0)
        return false;

    if (!S_ISDIR(statbuf.st_mode))
        return false;
#endif // _WIN32

  return true;
}

/* ---------------------------------------------------------------- */

bool
file_exists (char const* pathname)
{
#ifdef _WIN32
    struct _stat statbuf;
    if (::_stat(pathname, &statbuf) < 0)
        return false;

    if (!(statbuf.st_mode & _S_IFREG))
        return false;
#else // _WIN32
    struct stat statbuf;
    if (::stat(pathname, &statbuf) < 0)
        return false;

    if (!S_ISREG(statbuf.st_mode))
        return false;
#endif // _WIN32

    return true;
}

/* ---------------------------------------------------------------- */

char*
get_cwd (char* buf, size_t size)
{
#ifdef _WIN32
  return ::_getcwd(buf, size);
#else // _WIN32
  return ::getcwd(buf, size);
#endif // _WIN32
}

/* ---------------------------------------------------------------- */

bool
set_cwd (char const* pathname)
{
#ifdef _WIN32
    return ::_chdir(pathname) >= 0;
#else // _WIN32
    return ::chdir(pathname) >= 0;
#endif // _WIN32
}

/* ---------------------------------------------------------------- */

bool
mkdir (char const* pathname/*, mode_t mode*/)
{
#ifdef _WIN32
    if (::_mkdir(pathname) < 0)
        return false;
#else // _WIN32
    if (::mkdir(pathname, S_IRWXU | S_IRGRP | S_IXGRP) < 0)
        return false;
#endif // _WIN32

  return true;
}

/* ---------------------------------------------------------------- */

bool
rmdir (char const* pathname)
{
#ifdef _WIN32
        return ::_rmdir(pathname) >= 0;
#else // _WIN32
        return ::rmdir(pathname) >= 0;
#endif // _WIN32
}
/* ---------------------------------------------------------------- */

bool
unlink (char const* pathname)
{
#ifdef _WIN32
    return ::_unlink(pathname) >= 0;
#else // _WIN32
    return ::unlink(pathname) >= 0;
#endif // _WIN32
}

/* ---------------------------------------------------------------- */

char const*
get_app_data_dir (void)
{
    if (*app_data_path != 0)
        return app_data_path;

    // TODO: Use environment variables?

#ifdef _WIN32
    // SHGetFolderPathA seems to expect non-wide chars
    // http://msdn.microsoft.com/en-us/library/bb762181(VS.85).aspx
    // FIXME: Max length of home path?
    if (!SUCCEEDED(::SHGetFolderPathA(0, CSIDL_APPDATA, 0, 0, app_data_path)))
        throw util::Exception("Cannot determine home directory");
#else // _WIN32
    std::string p = join_path(get_home_dir(), ".local/share");
    if (p.size() >= PATH_MAX)
        throw util::Exception("Cannot determine home directory");
    std::strncpy(app_data_path, p.c_str(), PATH_MAX - 1);
    app_data_path[PATH_MAX - 1] = '\0';
#endif // _WIN32

  return app_data_path;
}

/* ---------------------------------------------------------------- */

char const*
get_home_dir (void)
{
    if (*home_path != 0)
        return home_path;

    // TODO: Use HOME environment variable?

#ifdef _WIN32
    // SHGetFolderPathA seems to expect non-wide chars
    // http://msdn.microsoft.com/en-us/library/bb762181(VS.85).aspx
    // FIXME: Max length of home path?
    if (!SUCCEEDED(::SHGetFolderPathA(0, CSIDL_APPDATA, 0, 0, home_path)))
        throw util::Exception("Cannot determine home directory");
#else // _WIN32
    uid_t user_id = ::geteuid();
    struct passwd* user_info = ::getpwuid(user_id);
    if (user_info == nullptr || user_info->pw_dir == nullptr)
        throw util::Exception("Cannot determine home directory");
    std::strncpy(home_path, user_info->pw_dir, PATH_MAX - 1);
    home_path[PATH_MAX - 1] = '\0';
#endif // _WIN32

  return home_path;
}

/* ---------------------------------------------------------------- */

bool
rename (char const* from, char const* to)
{
    if (std::rename(from, to) < 0)
        return false;

    return true;
}

/* ---------------------------------------------------------------- */

void
copy_file (char const* src, char const* dst)
{
    std::ifstream src_stream(src, std::ios::binary);
    if (!src_stream.good())
        throw util::FileException(src, std::strerror(errno));
    std::ofstream dst_stream(dst, std::ios::binary);
    if (!dst_stream.good())
        throw util::FileException(dst, std::strerror(errno));

    int const BUFFER_SIZE = 4096;
    char buffer[BUFFER_SIZE];
    while (!src_stream.eof())
    {
        src_stream.read(buffer, BUFFER_SIZE);
        if (src_stream.bad())
            throw util::FileException(src, std::strerror(errno));
        dst_stream.write(buffer, src_stream.gcount());
        if (!dst_stream.good())
            throw util::FileException(dst, std::strerror(errno));
    }

    src_stream.close();
    if (src_stream.bad())
        throw util::FileException(src, std::strerror(errno));
    dst_stream.close();
    if (!dst_stream.good())
        throw util::FileException(dst, std::strerror(errno));
}

/* ---------------------------------------------------------------- */

std::string
get_cwd_string (void)
{
    std::size_t size = 1 << 8;
    while (true)
    {
        char* buf = new char[size];
        if (get_cwd(buf, size) == buf)
        {
            std::string ret(buf);
            delete[] buf;
            return ret;
        }
        delete[] buf;
        size = size << 1;
        if (size > (1 << 15))
            throw Exception("Error storing CWD");
    }
}

/* ---------------------------------------------------------------- */
// TODO Test on Win32 and OSX

std::string
get_binary_path (void)
{
    char path[PATH_MAX];
    std::fill(path, path + PATH_MAX, '\0');

#if defined(_WIN32)

#   ifdef UNICODE
#       error "Unicode defined but not supported"
#   endif

    TCHAR omgwtf[PATH_MAX];
    int n_chars = GetModuleFileName(nullptr, omgwtf, PATH_MAX);
    std::copy(omgwtf, omgwtf + n_chars, path);

#elif defined(__APPLE__)

    /*
     * http://developer.apple.com/library/mac/#documentation/
     * ...    Darwin/Reference/ManPages/man3/dyld.3.html
     */
    uint32_t pathmax = PATH_MAX;
    int success = _NSGetExecutablePath(path, &pathmax);
    int n_chars = 0;
    if (success < 0)
        throw std::runtime_error(
            "Could not determine binary path: _NSGetExecutablePath failed!");
    else
    {
        char real[PATH_MAX];
        if (::realpath(path, real) == nullptr)
            throw std::runtime_error(
                "Could not determine binary path: realpath failed!");
        ::strncpy(path, real, PATH_MAX);
    }

#elif defined(__linux) || defined(__CYGWIN__)

    ssize_t n_chars = ::readlink("/proc/self/exe", path, PATH_MAX);

#else
#   error "Cannot determine binary path: Unsupported OS"
#endif

    if (n_chars >= PATH_MAX)
        throw std::runtime_error("Buffer size too small!");

    return std::string(path);
}

/* ---------------------------------------------------------------- */

bool
is_absolute (std::string const& path)
{
#ifdef _WIN32
    return path.size() >= 2 && std::isalpha(path[0]) && path[1] == ':';
#else
    return path.size() >= 1 && path[0] == '/';
#endif
}

std::string
sanitize_path (std::string const& path)
{
    if (path.empty())
        return "";

    std::string result = path;

    /* Replace backslashes with slashes. */
    std::replace(result.begin(), result.end(), '\\', '/');

    /* Remove double slashes. */
    for (std::size_t i = 0; i < result.size() - 1; )
    {
        if (result[i] == '/' && result[i + 1] == '/')
            result.erase(i, 1);
        else
            i += 1;
    }

    /* Remove trailing slash if result != "/". */
    if (result.size() > 1 && result[result.size() - 1] == '/')
        result.erase(result.end() - 1);

    return result;
}

std::string
join_path (std::string const& path1, std::string const& path2)
{
    std::string p2 = sanitize_path(path2);
    if (is_absolute(p2))
        return p2;

#ifdef _WIN32
    if (!p2.empty() && p2[0] == '/')
        return sanitize_path(path1) + p2;
#endif

    return sanitize_path(path1) + '/' + p2;
}

std::string
abspath (std::string const& path)
{
    return join_path(get_cwd_string(), path);
}

std::string
dirname (std::string const& path)
{
    if (path.empty())
        return ".";

    /* Skip group of slashes at the end. */
    std::size_t len = path.size();
    while (len > 0 && path[len - 1] == '/')
        len -= 1;
    if (len == 0)
        return "/";

    /* Skip basename. */
    while (len > 0 && path[len - 1] != '/')
        len -= 1;
    if (len == 0)
        return ".";

    /* Skip group of slashes. */
    while (len > 1 && path[len - 1] == '/')
        len -= 1;

    return path.substr(0, len);
}

std::string
basename (std::string const& path)
{
    /* Skip group of slashes at the end. */
    std::size_t len = path.size();
    while (len > 0 && path[len - 1] == '/')
        len -= 1;
    if (len == 0)
        return "";

    /* Skip basename. */
    std::size_t base = len - 1;
    while (base > 0 && path[base - 1] != '/')
        base -= 1;

    return path.substr(base, len - base);
}

/* ---------------------------------------------------------------- */

std::string
replace_extension (std::string const& fn, std::string const& ext)
{
    std::size_t slashpos = fn.find_last_of('/');
    if (slashpos == std::string::npos)
        slashpos = 0;

    std::size_t dotpos = fn.find_last_of('.');
    if (dotpos == std::string::npos || dotpos < slashpos)
        return fn + "." + ext;

    return fn.substr(0, dotpos) + "." + ext;
}

/*
 * ----------------------------- File IO  ----------------------------
 */

void
read_file_to_string (std::string const& filename, std::string* data)
{
    std::ifstream in(filename.c_str(), std::ios::binary);
    if (!in.good())
        throw util::FileException(filename, std::strerror(errno));
    in.seekg(0, std::ios::end);
    std::size_t length = in.tellg();
    in.seekg(0, std::ios::beg);
    data->resize(length);
    in.read(&(*data)[0], length);
    in.close();
}

void
write_string_to_file (std::string const& data, std::string const& filename)
{
    write_string_to_file(&data[0], data.size(), filename);
}

void
write_string_to_file (char const* data, std::size_t len,
    std::string const& filename)
{
    std::ofstream out(filename.c_str(), std::ios::binary);
    if (!out.good())
        throw util::FileException(filename, std::strerror(errno));
    out.write(data, len);
    out.close();
}

UTIL_FS_NAMESPACE_END
UTIL_NAMESPACE_END

/* ----------------------------------------------------------------
 * ----------------- File and Directory abstraction ---------------
 * ---------------------------------------------------------------- */

UTIL_NAMESPACE_BEGIN
UTIL_FS_NAMESPACE_BEGIN

std::string
File::get_absolute_name (void) const
{
#ifdef _WIN32
    // FIXME: Use '/' even on windows?
    return (!path.empty() && *path.rbegin() == '\\'
        ? path + name
        : path + "\\" + name);
#else
    return (!path.empty() && *path.rbegin() == '/'
        ? path + name
        : path + "/" + name);
#endif
}

/* ---------------------------------------------------------------- */

bool
File::operator< (File const& rhs) const
{
    if (this->is_dir && !rhs.is_dir)
        return true;
    else if (!this->is_dir && rhs.is_dir)
        return false;
    else if (this->path < rhs.path)
        return true;
    else if (rhs.path < this->path)
        return false;
    else
        return (this->name < rhs.name);
}

/* ---------------------------------------------------------------- */

void
Directory::scan (std::string const& path)
{
    this->clear();

#ifdef _WIN32
    WIN32_FIND_DATA data;
    HANDLE hf = FindFirstFile((path + "/*").c_str(), &data);
    if (hf == INVALID_HANDLE_VALUE)
        throw Exception("Cannot open directory");

    do
    {
        if (!std::strcmp(data.cFileName, "."))
            continue;
        if (!std::strcmp(data.cFileName, ".."))
            continue;

        this->push_back(File());
        this->back().path = sanitize_path(path);
        this->back().name = data.cFileName;
        this->back().is_dir =
            (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
    }
    while (FindNextFile(hf, &data) != 0);

    FindClose(hf);
#else
    DIR *dp = ::opendir(path.c_str());
    if (dp == nullptr)
        throw Exception("Cannot open directory: ", std::strerror(errno));

    struct dirent *ep;
    while ((ep = ::readdir(dp)))
    {
        if (!std::strcmp(ep->d_name, "."))
            continue;
        if (!std::strcmp(ep->d_name, ".."))
            continue;
        this->push_back(File());
        this->back().path = path;
        this->back().name = ep->d_name;
        this->back().is_dir = (ep->d_type == DT_DIR);
        if (ep->d_type == DT_UNKNOWN)
        {
            struct stat path_stat;
            if (::stat(join_path(path, ep->d_name).c_str(), &path_stat) >= 0)
                this->back().is_dir = S_ISDIR(path_stat.st_mode);
        }
    }
    ::closedir(dp);
#endif
}

/*
 * --------------------- File locking mechanism ----------------------
 */

FileLock::FileLock (std::string const& filename)
{
    switch (this->acquire_retry(filename))
    {
        case LOCK_CREATED: break;
        default: throw util::Exception(this->reason);
    }
}

FileLock::Status
FileLock::acquire (std::string const& filename)
{
    /* Check if lock file exists. */
    this->lockfile = filename + ".lock";
    this->reason.clear();
    if (fs::file_exists(this->lockfile.c_str()))
    {
        this->reason = "Previous lock existing";
        return FileLock::LOCK_EXISTS;
    }

    /* Finally create the lock file. */
    std::ofstream touch(this->lockfile.c_str(), std::ios::binary);
    if (!touch.good())
    {
        this->reason = "Error locking: ";
        this->reason += std::strerror(errno);
        return FileLock::LOCK_CREATE_ERROR;
    }
    touch.close();

    /* Return success, lock is created. */
    return FileLock::LOCK_CREATED;
}

FileLock::Status
FileLock::acquire_retry (std::string const& filename, int retries, int sleep)
{
    /* Try to acquire file lock for 'retry' times. */
    while (retries > 0)
    {
        Status status = this->acquire(filename);
        if (status != FileLock::LOCK_EXISTS)
            return status;

        system::sleep(sleep);
        retries -= 1;

        /* Fail if all retries have been used. */
        if (retries <= 0)
        {
            this->reason = "Previous lock persisting";
            return FileLock::LOCK_PERSISTENT;
        }
    }

    /* Return success, lock is created. */
    return FileLock::LOCK_CREATED;
}

bool
FileLock::is_locked (std::string const& filename)
{
    std::string lockfname = filename + ".lock";
    return fs::file_exists(lockfname.c_str());
}

bool
FileLock::wait_lock (std::string const& filename, int retries, int sleep)
{
    while (retries > 0 && this->is_locked(filename))
    {
        system::sleep(sleep);
        retries -= 1;
        if (retries <= 0)
            return false;
    }
    return true;
}

bool
FileLock::release (void)
{
    if (this->lockfile.empty())
        return false;
    return fs::unlink(this->lockfile.c_str());
}

UTIL_FS_NAMESPACE_END
UTIL_NAMESPACE_END
