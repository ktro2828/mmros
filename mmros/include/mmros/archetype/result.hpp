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

#ifndef MMROS__ARCHETYPE__RESULT_HPP_
#define MMROS__ARCHETYPE__RESULT_HPP_

#include <exception>
#include <stdexcept>
#include <string>
#include <variant>

namespace mmros
{
/**
 * @brief An enumerate to represent error kind.
 */
enum InferenceError_t {
  TENSORRT,  //!< TensorRT related error.
  CUDA,      //!< CUDA related error.
  UNKNOWN,   //!< Unknown error.
};

/**
 * @brief A class to hold error kind and message.
 */
struct InferenceError
{
  /**
   * @brief Construct a new InferenceError object with message.
   *
   * @param kind Error kind.
   * @param msg Error message.
   */
  InferenceError(const InferenceError_t & kind, const std::string & msg) : kind(kind), msg(msg) {}

  /**
   * @brief Construct a new InferenceError object without any message.
   *
   * @param kind Error kind.
   */
  explicit InferenceError(const InferenceError_t & kind) : kind(kind), msg("") {}

  InferenceError_t kind;  //!< Error kind.
  std::string msg;        //!< Error message.
};

/**
 * @brief An exception class for `InferenceError`.
 */
class InferenceException : public std::exception
{
public:
  /**
   * @brief Construct a new InferenceException object.
   *
   * @param error `InferenceError` object.
   */
  explicit InferenceException(const InferenceError & error) : error_(error)
  {
    switch (error_.kind) {
      case InferenceError_t::TENSORRT:
        msg_ = "[TENSORRT]: " + error_.msg;
        break;
      case InferenceError_t::CUDA:
        msg_ = "[CUDA]: " + error_.msg;
        break;
      default:
        msg_ = "[UNKNOWN]: " + error_.msg;
    }
  }

  /**
   * @brief Return the error message.
   */
  const char * what() const throw() { return msg_.c_str(); }

private:
  InferenceError error_;  //!< Error object.
  std::string msg_;       //!< Error message.
};

/**
 * @brief An class to hold expected value or error.
 *
 * @tparam T Data type of expected value.
 */
template <typename T>
class Result
{
public:
  /**
   * @brief Construct a new Result object with an expected value.
   *
   * @param value Expected value.
   */
  explicit Result(const T & value) : value_(value) {}

  /**
   * @brief Construct a new Result object with an error.
   *
   * @param error `InferenceError` object.
   */
  explicit Result(const InferenceError & error) : value_(error) {}

  /**
   * @brief Check whether holding value is expected type.
   */
  bool isOk() const noexcept { return std::holds_alternative<T>(value_); }

  /**
   * @brief Return the expected value if it holds, otherwise throw `InferenceException`.
   */
  T unwrap() const
  {
    if (isOk()) {
      return std::get<T>(value_);
    } else {
      throw InferenceException(std::get<InferenceError>(value_));
    }
  }

private:
  std::variant<T, InferenceError> value_;  //!< Container of expected value or error.
};

/**
 * @brief Returns `Result` with an expected value.
 *
 * @tparam T Data type of the expected value.
 * @param value Expected value.
 */

template <typename T>
Result<T> Ok(const T & value) noexcept
{
  return Result<T>(value);
}

/**
 * @brief Return `Result` with en error.
 *
 * @tparam T Data type of the expected value.
 * @param error `InferenceError` object.
 */
template <typename T>
Result<T> Err(const InferenceError & error) noexcept
{
  return Result<T>(error);
}

/**
 * @brief Return `Result` with en error.
 *
 * @tparam T Data type of the expected value.
 * @param kind Error kind.
 */
template <typename T>
Result<T> Err(const InferenceError_t & kind) noexcept
{
  InferenceError error(kind);
  return Result<T>(error);
}

/**
 * @brief Return `Result` with en error.
 *
 * @tparam T Data type of the expected value.
 * @param kind Error kind.
 * @param msg Error message.
 */
template <typename T>
Result<T> Err(const InferenceError_t & kind, const std::string & msg) noexcept
{
  InferenceError error(kind, msg);
  return Result<T>(error);
}
}  // namespace mmros
#endif  // MMROS__ARCHETYPE__RESULT_HPP_
