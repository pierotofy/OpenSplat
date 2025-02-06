# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.22.1)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(nanoflann-populate NONE)



include(ExternalProject)
ExternalProject_Add(nanoflann-populate
                     "UPDATE_DISCONNECTED" "False" "URL" "https://github.com/jlblancoc/nanoflann/archive/refs/tags/v1.5.5.zip"
                    SOURCE_DIR          "/home/evanokeeffe/Modelling/OpenSplat/my_build/_deps/nanoflann-src"
                    BINARY_DIR          "/home/evanokeeffe/Modelling/OpenSplat/my_build/_deps/nanoflann-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
)


