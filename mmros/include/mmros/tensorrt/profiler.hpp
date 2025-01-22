// Copyright 2023 TIER IV, Inc.
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

#ifndef MMROS__TENSORRT__PROFILER_HPP_
#define MMROS__TENSORRT__PROFILER_HPP_

#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <ostream>
#include <string>
#include <vector>

namespace mmros::tensorrt
{
/**
 * @brief Per-layer information of a convolution layer.
 */
struct ConvLayerInfo
{
  int32_t in_c;              //!< Input channel.
  int32_t out_c;             //!< Output channel.
  int32_t w;                 //!< Width.
  int32_t h;                 //!< Height.
  int32_t k;                 //!< Kernel size.
  int32_t stride;            //!< Stride.
  int32_t groups;            //!< Number of groups.
  nvinfer1::LayerType type;  //!< Layer type.
};

/**
 * @brief Collect per-layer profile information, assuming times are reported in
 * the same order.
 */
class Profiler : public nvinfer1::IProfiler
{
public:
  /**
   * @brief Record of layer profile information.
   */
  struct Record
  {
    float time{0.0f};
    size_t count{0};
    float min_time{-1.0f};
    size_t index;
  };

  /**
   * @brief Construct a new Profiler object.
   *
   * @param profilers Source profilers to merge.
   */
  explicit Profiler(const std::vector<Profiler> & profilers = {});

  /**
   * @brief Report the layer time.
   *
   * @param layer_name Name of the layer.
   * @param ms Time in milliseconds.
   */
  void reportLayerTime(const char * layer_name, float ms) noexcept final;

  /**
   * @brief Set per-Layer information of the model.
   *
   * @param layer Layer information.
   */
  void setLayerProfile(const nvinfer1::ILayer * layer) noexcept;

  /**
   * @brief Get a printable respresentation of Profiler.
   *
   * @return std::string
   */
  [[nodiscard]] std::string toString() const;

private:
  std::map<std::string, Record> records_;             //!< Records of the each layer.
  std::map<std::string, ConvLayerInfo> conv_layers_;  //!< Convolution layer information.
  size_t index_;                                      //!< Index of the layer.
};

/**
 * @brief Output Profiler to ostream.
 *
 * @param os Output stream
 * @param profiler Profiler to output.
 * @return std::ostream& Output stream.
 */
std::ostream & operator<<(std::ostream & os, const Profiler & profiler);
}  // namespace mmros::tensorrt
#endif  // MMROS__TENSORRT__PROFILER_HPP_
