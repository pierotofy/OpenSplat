/*
 * Copyright (C) 2015, Simon Fuhrmann, Andre Schulz
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef UTIL_FS_HEADER
#define UTIL_FS_HEADER

#include <string>
#include <vector>

#include "util/defines.h"

UTIL_NAMESPACE_BEGIN
UTIL_FS_NAMESPACE_BEGIN

/*
 * -------------------------- Path operations ------------------------
 */

/** Determines if the given path is a directory. */
bool exists (char const* pathname);

/** Determines if the given path is a directory. */
bool dir_exists (char const* pathname);

/** Determines if the given path is a file. */
bool file_exists (char const* pathname);

/** Determines the current user's path for application data. */
char const* get_app_data_dir (void);

/** Determines the home path for the current user. */
char const* get_home_dir (void);

/** Determines the current working directory of the process. */
char* get_cwd (char* buf, std::size_t size);

/**
 * Changes the current working directory to 'pathname' and returns true
 * on success. NOTE: An application should never change its working
 * directory. Make sure this is really necessary.
 */
bool set_cwd (char const* pathname);

/*
 * -------------------------- File operations ------------------------
 */

// TODO: define modes for mkdir() for all platforms?
/** Creates a new directory. */
bool mkdir (char const* pathname/*, mode_t mode*/);

/** Removes an empty directory */
bool rmdir (char const* pathname);

/** Unlinks (deletes) the given file. */
bool unlink (char const* pathname);

/** Renames the given file 'from' to new name 'to'. */
bool rename (char const* from, char const* to);

/** Copies a file from 'src' to 'dst', throws FileException on error. */
void copy_file (char const* src, char const* dst);

/*
 * ----------------------------- File IO  ----------------------------
 */

/** Reads the whole file into a string. Throws on error. */
void
read_file_to_string (std::string const& filename, std::string* data);

/** Writes the given data into a file. Throws on error. */
void
write_string_to_file (std::string const& data, std::string const& filename);

/** Writes the given data into a file. Throws on error. */
void
write_string_to_file (char const* data, std::size_t len,
    std::string const& filename);

/*
 * ------------------------- String processing -----------------------
 */

/** Determines the CWD and returns a convenient string. */
std::string get_cwd_string (void);

/** Returns the path of the binary currently executing. */
std::string get_binary_path (void);

/** Checks whether the given path is absolute. */
bool is_absolute (std::string const& path);

/** Canonicalize slashes in the given path. */
std::string sanitize_path (std::string const& path);

/**
 * Concatenate and canonicalize two paths. The result is where you would
 * end up after cd path1; cd path2.
 */
std::string join_path (std::string const& path1, std::string const& path2);

/** Returns the absolute representation of the given path. */
std::string abspath (std::string const& path);

/** Returns the directory name component of the given path. */
std::string dirname (std::string const& path);

/** Returns the file name component of the given path. */
std::string basename (std::string const& path);

/**
 * Replaces extension of the given file with 'ext'. If the file name
 * does not have an extension, the given extension is appended.
 */
std::string replace_extension (std::string const& fn, std::string const& ext);

/*
 * ------------------------- File abstraction ------------------------
 */

struct File
{
    std::string path;
    std::string name;
    bool is_dir;

    File (void);
    File (std::string const& path, std::string const& name, bool isdir = false);
    std::string get_absolute_name (void) const;
    bool operator< (File const& rhs) const;
};

/*
 * ------------------------- Directory reading -----------------------
 */

/** Directory abstraction to scan directory contents. */
class Directory : public std::vector<File>
{
public:
    Directory (void);
    Directory (std::string const& path);
    void scan (std::string const& path);
};

/*
 * --------------------- File locking mechanism ----------------------
 */

/**
 * A simple file-based file lock implementation.
 * A file NAME.lock is created when a lock is acquired, and
 * the file is removed when the lock is released. If the lock
 * already exist, 'acquire' returns false. 'acquire_retry' retries for
 * a few seconds and finally gives up returning false. Note that the ".lock"
 * suffix is automatically used and must not be part of the given filenames.
 */
class FileLock
{
public:
    enum Status
    {
        /** The lock has been created successfully. */
        LOCK_CREATED,
        /** The lock has NOT been created because a lock already exists. */
        LOCK_EXISTS,
        /** The lock has NOT been created because an existing lock persisted. */
        LOCK_PERSISTENT,
        /** The lock has NOT been created because of file system issues. */
        LOCK_CREATE_ERROR
    };

public:
    FileLock (void);

    /**
     * Acquires a lock for the given filename. If the lock already exists,
     * the operation is re-attempted using default values. If the lock
     * cannot be created an exception is thrown.
     */
    FileLock (std::string const& filename);

    /** Removes the lock if it exists. */
    ~FileLock (void);

    /**
     * Tries to acquire a lock for the given filename and returns a status.
     */
    Status acquire (std::string const& filename);

    /**
     * Tries to acquire a lock for the given filename.
     * If a lock exists, the operation is re-attempted 'retry' times
     * with 'sleep' milli seconds delay between the attempts.
     */
    Status acquire_retry (std::string const& filename,
        int retries = 50, int sleep = 100);

    /**
     * Returns true if a lock for the given filename exists.
     */
    bool is_locked (std::string const& filename);

    /**
     * Waits until a lock for given filename is released. If filename is
     * not locked, the method returns true immediately. If the lock is not
     * released within the specified bounds, the method returns false.
     */
    bool wait_lock (std::string const& filename,
        int retries = 50, int sleep = 100);

    /**
     * Removes the lock if it exists.
     * If removing the lock fails, the method returns false.
     */
    bool release (void);

    /**
     * If locking failes, this returns the reason for failure.
     */
    std::string const& get_reason (void) const;

private:
    std::string lockfile;
    std::string reason;
};

/*
 * -------------------------- Implementation -------------------------
 */

inline
File::File (void)
    : is_dir(false)
{
}

inline
File::File (std::string const& path, std::string const& name, bool isdir)
    : path(path), name(name), is_dir(isdir)
{
}

inline
Directory::Directory (void)
{
}

inline
Directory::Directory (std::string const& path)
{
    this->scan(path);
}

inline
FileLock::FileLock (void)
{
}

inline
FileLock::~FileLock (void)
{
    this->release();
}

inline std::string const&
FileLock::get_reason (void) const
{
    return this->reason;
}

UTIL_FS_NAMESPACE_END
UTIL_NAMESPACE_END

#endif /* UTIL_FS_HEADER */
