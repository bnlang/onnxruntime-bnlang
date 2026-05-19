#pragma once

#include "bnl/plugin.h"
#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <vector>

namespace ortb::tensor
{

    const char *dtype_to_string(ONNXTensorElementDataType t);
    struct InputBuffer
    {
        Ort::Value value{nullptr};
        std::vector<int64_t> shape;
        std::vector<float> f32;
        std::vector<double> f64;
        std::vector<Ort::Float16_t> f16;
        std::vector<int64_t> i64;
        std::vector<int32_t> i32;
        std::vector<uint8_t> u8;
        std::vector<uint8_t> b8;
    };

    std::unique_ptr<InputBuffer> from_bnl(const bnl_api *api,
                                          bnl_value *spec,
                                          const std::string &input_name);

    bnl_value *to_bnl_handle(const bnl_api *api, Ort::Value &&v);

    bnl_value *tensor_to_data(const bnl_api *api, int handle);

    void tensor_close(int handle);

    int tensor_argmax_last(int handle, int vocab);

    int tensor_sample_last(int handle, int vocab,
                           double temperature, int top_k,
                           double top_p, double seed);

    void register_tensor_natives(const bnl_api *api, bnl_module *mod);

} // namespace ortb::tensor
