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
#include <optional>
#include <stdexcept>

namespace mmros
{
/**
 * @brief Custom exception
 */
class InferenceException : public std::exception
{
public:
  const char * what() const throw() { return "TensorRT inference failure."; }
};

template <typename T>
class Result
{
public:
  explicit Result(const T & value) : value_(value) {}

  Result() : value_(std::nullopt) {}

  bool isOk() const noexcept { return value_.has_value(); }

  T unwrap() const
  {
    if (isOk()) {
      return value_.value();
    } else {
      throw InferenceException();
    }
  }

private:
  std::optional<T> value_;
};

template <typename T>
Result<T> Ok(const T & value)
{
  return Result<T>(value);
}
}  // namespace mmros
#endif  // MMROS__ARCHETYPE__RESULT_HPP_
