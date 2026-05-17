#pragma once

#include "bnl/plugin.h"
#include <onnxruntime_cxx_api.h>

namespace ortb::session_options
{
    void apply(const bnl_api *api, bnl_value *map, Ort::SessionOptions &out);
} // namespace ortb::session_options
