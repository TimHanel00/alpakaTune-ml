# SPDX-License-Identifier: MPL-2.0
cmake_minimum_required(VERSION 3.25)

project(alpakaTuneMLDependencies LANGUAGES NONE)

set(
    ALPAKATUNE_ML_ALPAKATUNE_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/alpaka3-tuner"
    CACHE PATH
    "Pinned alpaka3-tuner submodule checkout"
)
set(
    ALPAKATUNE_ML_ALPAKATUNE_BUILD_DIR
    "${CMAKE_BINARY_DIR}/alpaka3-tuner"
    CACHE PATH
    "Nested alpaka3-tuner build directory"
)
set(
    ALPAKATUNE_ML_ALPAKATUNE_GIT_REPOSITORY
    "https://github.com/TimHanel00/alpaka3-tuner.git"
    CACHE STRING
    "Canonical alpaka3-tuner repository"
)
option(ALPAKATUNE_ML_ENABLE_CUDA "Enable CUDA in the nested alpaka3-tuner build" ON)
option(ALPAKATUNE_ML_BUILD_ALPAKATUNE_TESTS "Build nested alpakaTune tests" OFF)
option(ALPAKATUNE_ML_BUILD_ALPAKATUNE_HEADER_CHECKS "Build nested header checks" OFF)

cmake_path(
    ABSOLUTE_PATH ALPAKATUNE_ML_ALPAKATUNE_SOURCE_DIR
    BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    NORMALIZE
    OUTPUT_VARIABLE _alpakatune_source
)
cmake_path(
    ABSOLUTE_PATH ALPAKATUNE_ML_ALPAKATUNE_BUILD_DIR
    BASE_DIRECTORY "${CMAKE_BINARY_DIR}"
    NORMALIZE
    OUTPUT_VARIABLE _alpakatune_build
)

if(NOT EXISTS "${_alpakatune_source}/CMakeLists.txt")
    message(
        FATAL_ERROR
        "alpaka3-tuner is not initialized at ${_alpakatune_source}. "
        "Run: git submodule update --init --recursive"
    )
endif()

# The parent repository's gitlink is the source-of-truth pin. Require the
# checked-out submodule to match it, then persist the exact revision with every
# generated campaign so a dataset never depends on an implicit branch tip.
find_package(Git REQUIRED)
execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_CURRENT_SOURCE_DIR}" rev-parse HEAD:alpaka3-tuner
    RESULT_VARIABLE _alpakatune_revision_result
    OUTPUT_VARIABLE _alpakatune_revision
    ERROR_VARIABLE _alpakatune_revision_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(LENGTH "${_alpakatune_revision}" _alpakatune_revision_length)
if(
    NOT _alpakatune_revision_result EQUAL 0
    OR NOT _alpakatune_revision MATCHES "^[0-9a-f]+$"
    OR NOT _alpakatune_revision_length EQUAL 40
)
    message(
        FATAL_ERROR
        "Cannot resolve the alpaka3-tuner gitlink revision: ${_alpakatune_revision_error}"
    )
endif()
execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${_alpakatune_source}" rev-parse HEAD
    RESULT_VARIABLE _alpakatune_checkout_result
    OUTPUT_VARIABLE _alpakatune_checkout_revision
    ERROR_VARIABLE _alpakatune_checkout_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _alpakatune_checkout_result EQUAL 0)
    message(FATAL_ERROR "Cannot resolve the alpaka3-tuner checkout: ${_alpakatune_checkout_error}")
endif()
if(NOT _alpakatune_checkout_revision STREQUAL _alpakatune_revision)
    message(
        FATAL_ERROR
        "alpaka3-tuner checkout ${_alpakatune_checkout_revision} does not match "
        "the pinned gitlink ${_alpakatune_revision}. Run: git submodule update --init --recursive"
    )
endif()

# Configure the child before adding it. The ML repository needs its examples
# for data collection but never installs or vendors the runtime sources.
set(alpakaTune_BUILD_EXAMPLES ON CACHE BOOL "" FORCE)
set(alpakaTune_BUILD_TESTING ${ALPAKATUNE_ML_BUILD_ALPAKATUNE_TESTS} CACHE BOOL "" FORCE)
set(alpakaTune_HEADER_CHECKS ${ALPAKATUNE_ML_BUILD_ALPAKATUNE_HEADER_CHECKS} CACHE BOOL "" FORCE)
set(alpaka_DEP_CUDA ${ALPAKATUNE_ML_ENABLE_CUDA} CACHE BOOL "" FORCE)
add_subdirectory("${_alpakatune_source}" "${_alpakatune_build}")

set(_alpakatune_source_mode "submodule")
set(_alpakatune_metadata_dir "${CMAKE_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${_alpakatune_metadata_dir}")

function(_alpakatune_json_escape output value)
    set(_escaped "${value}")
    string(REPLACE "\\" "\\\\" _escaped "${_escaped}")
    string(REPLACE "\"" "\\\"" _escaped "${_escaped}")
    string(REPLACE "\n" "\\n" _escaped "${_escaped}")
    string(REPLACE "\r" "\\r" _escaped "${_escaped}")
    set(${output} "${_escaped}" PARENT_SCOPE)
endfunction()

function(_alpakatune_shell_quote output value)
    string(REPLACE "'" "'\"'\"'" _quoted "${value}")
    set(${output} "'${_quoted}'" PARENT_SCOPE)
endfunction()

_alpakatune_json_escape(ALPAKATUNE_PATH_SOURCE_JSON "${_alpakatune_source}")
_alpakatune_json_escape(ALPAKATUNE_PATH_BUILD_JSON "${_alpakatune_build}")
_alpakatune_json_escape(ALPAKATUNE_PATH_REPOSITORY_JSON "${ALPAKATUNE_ML_ALPAKATUNE_GIT_REPOSITORY}")
_alpakatune_json_escape(ALPAKATUNE_PATH_TAG_JSON "${_alpakatune_revision}")
_alpakatune_json_escape(ALPAKATUNE_PATH_MODE_JSON "${_alpakatune_source_mode}")
_alpakatune_json_escape(ALPAKATUNE_PATH_METADATA_JSON "${_alpakatune_metadata_dir}")

_alpakatune_shell_quote(ALPAKATUNE_PATH_SOURCE_SHELL "${_alpakatune_source}")
_alpakatune_shell_quote(ALPAKATUNE_PATH_BUILD_SHELL "${_alpakatune_build}")
_alpakatune_shell_quote(ALPAKATUNE_PATH_REPOSITORY_SHELL "${ALPAKATUNE_ML_ALPAKATUNE_GIT_REPOSITORY}")
_alpakatune_shell_quote(ALPAKATUNE_PATH_TAG_SHELL "${_alpakatune_revision}")
_alpakatune_shell_quote(ALPAKATUNE_PATH_MODE_SHELL "${_alpakatune_source_mode}")
_alpakatune_shell_quote(ALPAKATUNE_PATH_METADATA_SHELL "${_alpakatune_metadata_dir}")

set(ALPAKATUNE_PATH_SOURCE "${_alpakatune_source}")
set(ALPAKATUNE_PATH_BUILD "${_alpakatune_build}")
set(ALPAKATUNE_PATH_REPOSITORY "${ALPAKATUNE_ML_ALPAKATUNE_GIT_REPOSITORY}")
set(ALPAKATUNE_PATH_TAG "${_alpakatune_revision}")
set(ALPAKATUNE_PATH_MODE "${_alpakatune_source_mode}")
set(ALPAKATUNE_PATH_METADATA "${_alpakatune_metadata_dir}")

configure_file(
    cmake/alpakatune-paths.json.in
    "${_alpakatune_metadata_dir}/alpakatune-paths.json"
    @ONLY
)
configure_file(
    cmake/alpakatune-paths.sh.in
    "${_alpakatune_metadata_dir}/alpakatune-paths.sh"
    @ONLY
)
configure_file(
    cmake/alpakatune-paths.cmake.in
    "${_alpakatune_metadata_dir}/alpakatune-paths.cmake"
    @ONLY
)

message(STATUS "alpakaTune source (submodule): ${_alpakatune_source}")
message(STATUS "alpakaTune revision: ${_alpakatune_revision}")
message(STATUS "alpakaTune nested build: ${_alpakatune_build}")
message(STATUS "alpakaTune path metadata: ${_alpakatune_metadata_dir}/alpakatune-paths.json")
