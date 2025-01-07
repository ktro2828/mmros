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

#include "mmros/tensorrt/profiler.hpp"

#include <NvInfer.h>
#include <NvInferRuntimeBase.h>

#include <cstdint>
#include <iomanip>
#include <ostream>
#include <string>
#include <vector>

namespace mmros
{
Profiler::Profiler(const std::vector<Profiler> & profilers) : index_(0)
{
  for (const auto & profiler : profilers) {
    for (const auto & [name, record] : profiler.records_) {
      auto it = records_.find(name);
      if (it == records_.end()) {
        records_.emplace(name, record);
      } else {
        it->second.time += record.time;
        it->second.count += record.count;
      }
    }
  }
}

void Profiler::reportLayerTime(const char * layer_name, float ms) noexcept
{
  if (records_.count(layer_name) == 0) {
    return;
  }

  records_[layer_name].count++;
  records_[layer_name].time += ms;
  if (records_[layer_name].min_time == -1.0f) {
    records_[layer_name].min_time = ms;
    records_[layer_name].index = index_;
    ++index_;
  } else if (records_[layer_name].min_time > ms) {
    records_[layer_name].min_time = ms;
  }
}

void Profiler::setLayerProfile(const nvinfer1::ILayer * layer) noexcept
{
  if (const auto type = layer->getType(); type == nvinfer1::LayerType::kCONVOLUTION) {
    const auto name = layer->getName();

    // // TODO(ktro2828): Fix build error in dynamic_cast.
    // nvinfer1::IConvolutionLayer * conv = dynamic_cast<nvinfer1::IConvolutionLayer *>(layer);

    // nvinfer1::ITensor * in = layer->getInput(0);
    // nvinfer1::Dims in_dim = in->getDimensions();

    // nvinfer1::ITensor * out = layer->getOutput(0);
    // nvinfer1::Dims out_dim = out->getDimensions();

    // nvinfer1::Dims k_dims = conv->getKernelSizeNd();
    // nvinfer1::Dims s_dims = conv->getStrideNd();

    // int32_t kernel = k_dims.d[0];
    // int32_t stride = s_dims.d[0];
    // int32_t groups = conv->getNbGroups();

    // conv_layers_.insert_or_assign(
    //   name, ConvLayerInfo{
    //           in_dim.d[1], out_dim.d[1], in_dim.d[3], in_dim.d[2], kernel, stride, groups,
    //           type});
  }
}

std::string Profiler::toString() const
{
  std::ostringstream out;
  float total_time = 0.0;
  std::string layer_name = "Operation";

  int max_layer_name_length = static_cast<int>(layer_name.size());
  for (const auto & [name, record] : records_) {
    total_time += record.time;
    max_layer_name_length = std::max(max_layer_name_length, static_cast<int>(name.size()));
  }

  auto old_settings = out.flags();
  auto old_precision = out.precision();
  // Output header
  {
    out << "index, " << std::setw(12);
    out << std::setw(max_layer_name_length) << layer_name << " ";
    out << std::setw(12) << "Runtime"
        << "%,"
        << " ";
    out << std::setw(12) << "Invocations"
        << " , ";
    out << std::setw(12) << "Runtime[ms]"
        << " , ";
    out << std::setw(12) << "Avg Runtime[ms]"
        << " ,";
    out << std::setw(12) << "Min Runtime[ms]" << std::endl;
  }
  int index = index_;
  for (int i = 0; i < index; i++) {
    for (const auto & [name, record] : records_) {
      if (record.index == i) {
        out << i << ",   ";
        out << std::setw(max_layer_name_length) << name << ",";
        out << std::setw(12) << std::fixed << std::setprecision(1)
            << (record.time * 100.0F / total_time) << "%"
            << ",";
        out << std::setw(12) << record.count << ",";
        out << std::setw(12) << std::fixed << std::setprecision(2) << record.time << ", ";
        out << std::setw(12) << std::fixed << std::setprecision(2) << record.time / record.count
            << ", ";
        out << std::setw(12) << std::fixed << std::setprecision(2) << record.min_time << std::endl;
      }
    }
  }
  out.flags(old_settings);
  out.precision(old_precision);
  out << "========== total runtime = " << total_time << " ms ==========" << std::endl;

  return out.str();
}

std::ostream & operator<<(std::ostream & os, const Profiler * profiler)
{
  os << profiler->toString();
  return os;
}
}  // namespace mmros
