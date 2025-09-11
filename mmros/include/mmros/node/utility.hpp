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

#ifndef MMROS__NODE__UTILITY_HPP_
#define MMROS__NODE__UTILITY_HPP_

#include <rclcpp/rclcpp.hpp>

#include <optional>
#include <string>

namespace mmros::node
{
/**
 * @brief Get the QoS profile of a topic.
 *
 * @param node The node to query.
 * @param topic The topic to query.
 * @return std::optional<rclcpp::QoS> The QoS profile of the topic, or std::nullopt if the topic is
 * not published.
 */
inline std::optional<rclcpp::QoS> getTopicQos(const rclcpp::Node * node, const std::string & topic)
{
  const auto publisher_info = node->get_publishers_info_by_topic(topic);
  return publisher_info.size() != 1 ? std::nullopt
                                    : std::make_optional(publisher_info[0].qos_profile());
}

/**
 * @brief Resolve a topic name.
 *
 * @param node The node to query.
 * @param query The topic name to resolve.
 * @return std::string The resolved topic name.
 */
inline std::string resolveTopicName(rclcpp::Node * node, const std::string & query)
{
  return node->get_node_topics_interface()->resolve_topic_name(query);
}
}  // namespace mmros::node
#endif  // MMROS__NODE__UTILITY_HPP_
