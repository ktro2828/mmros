// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// Modified from:
// https://github.com/NVIDIA/TensorRT/blob/master/plugin/common/serialize.hpp

#ifndef MMROS__TENSORRT__PLUGIN__COMMON__TRT_SERIALIZE_HPP_
#define MMROS__TENSORRT__PLUGIN__COMMON__TRT_SERIALIZE_HPP_

#include <cassert>
#include <cstring>
#include <type_traits>
#include <vector>

namespace mmros::plugin
{
template <typename T>
inline void serialize_value(void ** buffer, T const & value);

template <typename T>
inline void deserialize_value(void const ** buffer, size_t * buffer_size, T * value);

namespace
{
template <typename T, class Enable = void>
struct Serializer
{
};

template <typename T>
struct Serializer<
  T, typename std::enable_if<
       std::is_arithmetic<T>::value || std::is_enum<T>::value || std::is_pod<T>::value>::type>
{
  static size_t serialized_size(T const & value) { return sizeof(T); }
  static void serialize(void ** buffer, T const & value)
  {
    ::memcpy(*buffer, &value, sizeof(T));
    reinterpret_cast<char *&>(*buffer) += sizeof(T);
  }
  static void deserialize(void const ** buffer, size_t * buffer_size, T * value)
  {
    assert(*buffer_size >= sizeof(T));
    ::memcpy(value, *buffer, sizeof(T));
    reinterpret_cast<char const *&>(*buffer) += sizeof(T);
    *buffer_size -= sizeof(T);
  }
};

template <>
struct Serializer<const char *>
{
  static size_t serialized_size(const char * value) { return strlen(value) + 1; }
  static void serialize(void ** buffer, const char * value)
  {
    ::strcpy(static_cast<char *>(*buffer), value);
    reinterpret_cast<char *&>(*buffer) += strlen(value) + 1;
  }
  static void deserialize(void const ** buffer, size_t * buffer_size, const char ** value)
  {
    *value = static_cast<char const *>(*buffer);
    size_t data_size = strnlen(*value, *buffer_size) + 1;
    assert(*buffer_size >= data_size);
    reinterpret_cast<char const *&>(*buffer) += data_size;
    *buffer_size -= data_size;
  }
};

template <typename T>
struct Serializer<
  std::vector<T>,
  typename std::enable_if<
    std::is_arithmetic<T>::value || std::is_enum<T>::value || std::is_pod<T>::value>::type>
{
  static size_t serialized_size(std::vector<T> const & value)
  {
    return sizeof(value.size()) + value.size() * sizeof(T);
  }
  static void serialize(void ** buffer, std::vector<T> const & value)
  {
    serialize_value(buffer, value.size());
    size_t nbyte = value.size() * sizeof(T);
    ::memcpy(*buffer, value.data(), nbyte);
    reinterpret_cast<char *&>(*buffer) += nbyte;
  }
  static void deserialize(void const ** buffer, size_t * buffer_size, std::vector<T> * value)
  {
    size_t size;
    deserialize_value(buffer, buffer_size, &size);
    value->resize(size);
    size_t nbyte = value->size() * sizeof(T);
    assert(*buffer_size >= nbyte);
    ::memcpy(value->data(), *buffer, nbyte);
    reinterpret_cast<char const *&>(*buffer) += nbyte;
    *buffer_size -= nbyte;
  }
};
}  // namespace

/**
 * @brief Computes the size required to serialize a given value in bytes.
 *
 * @tparam T The type of the object to be serialized.
 * @param[in] value The object to be computed its size.
 * @return size_t The number of bytes required to serialize @p value.
 */
template <typename T>
inline size_t serialized_size(T const & value)
{
  return Serializer<T>::serialized_size(value);
}

/**
 * @brief Serializes a given value into the provided buffer.
 *
 * This function uses @c Serializer<T> to encode @p value and store it in the memory pointed to by
 * @p buffer. The buffer points will be advanced by the number bytes written.
 *
 * @tparam T The type of the object to be serialized.
 * @param[in,out] buffer A pointer to the buffer where the serialized data will be written.
 * @param[in] value A constant reference to the object to be serialized.
 */
template <typename T>
inline void serialize_value(void ** buffer, T const & value)
{
  return Serializer<T>::serialize(buffer, value);
}

/**
 * @brief Deserializes a value from the provided buffer.
 *
 * This functions uses @c Serializer<T> to read data from @p buffer and reconstruct the object in @p
 * value. The buffer pointer and its size are updated accordingly to reflect the data consumed
 * during deserialization.
 *
 * @tparam T The type of the object to be deserialized.
 * @param[in,out] buffer A pointer to the buffer from which data will be read. This pointer is
 * updated to reflect the new position after deserialization.
 * @param[in,out] buffer_size A pointer to a variable tracking the remaining the size of the buffer.
 * @param[out] value A pointer to the object where the deserialized data will be stored.
 */
template <typename T>
inline void deserialize_value(void const ** buffer, size_t * buffer_size, T * value)
{
  return Serializer<T>::deserialize(buffer, buffer_size, value);
}
}  // namespace mmros::plugin
#endif  // MMROS__TENSORRT__PLUGIN__COMMON__TRT_SERIALIZE_HPP_
