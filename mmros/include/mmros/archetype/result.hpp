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

enum InferenceError_t {
  TENSORRT,  //!< TensorRT related error.
  CUDA,      //!< CUDA related error.
  UNKNOWN,   //!< Unknown error.
};

struct InferenceError
{
  InferenceError_t type;
  std::string msg;

  InferenceError(const InferenceError_t & type, const std::string & msg) : type(type), msg(msg) {}

  explicit InferenceError(const InferenceError_t & type) : type(type), msg("") {}
};

/**
 * @brief Custom exception
 */
class InferenceException : public std::exception
{
public:
  explicit InferenceException(const InferenceError & error) : error_(error)
  {
    switch (error_.type) {
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

  const char * what() const throw() { return msg_.c_str(); }

private:
  InferenceError error_;
  std::string msg_;
};

template <typename T>
class Result
{
public:
  explicit Result(const T & value) : value_(value) {}

  explicit Result(const InferenceError & error) : value_(error) {}

  bool isOk() const noexcept { return std::holds_alternative<T>(value_); }

  T unwrap() const
  {
    if (isOk()) {
      return std::get<T>(value_);
    } else {
      throw InferenceException(std::get<InferenceError>(value_));
    }
  }

private:
  std::variant<T, InferenceError> value_;
};

template <typename T>
Result<T> Ok(const T & value) noexcept
{
  return Result<T>(value);
}

template <typename T>
Result<T> Err(const InferenceError & error) noexcept
{
  return Result<T>(error);
}

template <typename T>
Result<T> Err(const InferenceError_t & type) noexcept
{
  InferenceError error(type);
  return Result<T>(error);
}

template <typename T>
Result<T> Err(const InferenceError_t & type, const std::string & msg) noexcept
{
  InferenceError error(type, msg);
  return Result<T>(error);
}
}  // namespace mmros
#endif  // MMROS__ARCHETYPE__RESULT_HPP_
