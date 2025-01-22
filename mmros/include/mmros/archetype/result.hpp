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

#include "mmros/archetype/exception.hpp"

#include <string>
#include <variant>

namespace mmros::archetype
{
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
   * @param error `MmRosError` object.
   */
  explicit Result(const MmRosError & error) : value_(error) {}

  /**
   * @brief Check whether holding value is expected type.
   */
  bool isOk() const noexcept { return std::holds_alternative<T>(value_); }

  /**
   * @brief Return the expected value if it holds, otherwise throw `MmRosException`.
   */
  T unwrap() const
  {
    if (isOk()) {
      return std::get<T>(value_);
    } else {
      throw MmRosException(std::get<MmRosError>(value_));
    }
  }

private:
  std::variant<T, MmRosError> value_;  //!< Container of expected value or error.
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
 * @param error `MmRosError` object.
 */
template <typename T>
Result<T> Err(const MmRosError & error) noexcept
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
Result<T> Err(const MmRosError_t & kind) noexcept
{
  MmRosError error(kind);
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
Result<T> Err(const MmRosError_t & kind, const std::string & msg) noexcept
{
  MmRosError error(kind, msg);
  return Result<T>(error);
}
}  // namespace mmros::archetype
#endif  // MMROS__ARCHETYPE__RESULT_HPP_
