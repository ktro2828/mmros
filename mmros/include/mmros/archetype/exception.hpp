// Copyright 2025 Kotaro Uetake.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MMROS__ARCHETYPE__EXCEPTION_HPP_
#define MMROS__ARCHETYPE__EXCEPTION_HPP_

#include <exception>
#include <string>

namespace mmros
{
/**
 * @brief An enumerate to represent error kind.
 */
enum MmRosError_t {
  TENSORRT,  //!< TensorRT related error.
  CUDA,      //!< CUDA related error.
  UNKNOWN,   //!< Unknown error.
};

/**
 * @brief A class to hold error kind and message.
 */
struct MmRosError
{
  /**
   * @brief Construct a new MmRosError object with message.
   *
   * @param kind Error kind.
   * @param msg Error message.
   */
  MmRosError(const MmRosError_t & kind, const std::string & msg) : kind(kind), msg(msg) {}

  /**
   * @brief Construct a new MmRosError object without any message.
   *
   * @param kind Error kind.
   */
  explicit MmRosError(const MmRosError_t & kind) : kind(kind), msg("") {}

  MmRosError_t kind;  //!< Error kind.
  std::string msg;    //!< Error message.
};

/**
 * @brief An exception class for `MmRosError`.
 */
class MmRosException : public std::exception
{
public:
  /**
   * @brief Construct a new MmRosException object.
   *
   * @param error `MmRosError` object.
   */
  explicit MmRosException(const MmRosError & error) : error_(error) { appendMessagePrefix(); }

  /**
   * @brief Construct a new MmRosException object with error kind and message.
   *
   * @param kind Error kind.
   * @param msg Error message.
   */
  MmRosException(const MmRosError_t & kind, const std::string & msg) : error_(kind, msg)
  {
    appendMessagePrefix();
  }

  /**
   * @brief Return the error message.
   */
  const char * what() const throw() { return msg_.c_str(); }

private:
  /**
   * @brief Append prefix to the error message depending on its kind.
   */
  void appendMessagePrefix() noexcept
  {
    switch (error_.kind) {
      case MmRosError_t::TENSORRT:
        msg_ = "[TENSORRT]: " + error_.msg;
        break;
      case MmRosError_t::CUDA:
        msg_ = "[CUDA]: " + error_.msg;
        break;
      default:
        msg_ = "[UNKNOWN]: " + error_.msg;
    }
  }

  MmRosError error_;  //!< Error object.
  std::string msg_;   //!< Error message.
};
}  // namespace mmros
#endif  // MMROS__ARCHETYPE__EXCEPTION_HPP_
