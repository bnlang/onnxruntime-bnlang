#include "session_options.h"

#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

#if defined(__has_include)
#if __has_include(<dml_provider_factory.h>)
#include <dml_provider_factory.h>
#define ORTB_HAS_DML 1
#endif
#endif

namespace ortb::session_options
{
    namespace
    {

        double get_number_or_throw(const bnl_api *api, bnl_value *map, const char *key)
        {
            bnl_value *v = api->map_get(api, map, key);
            if (api->get_type(v) != BNL_TYPE_NUMBER)
                throw std::runtime_error(std::string("session option '") + key + "' must be a number");
            return api->get_number(v);
        }

        void append_provider(const std::string &name,
                             const std::unordered_map<std::string, std::string> &options,
                             Ort::SessionOptions &opts)
        {
            std::vector<const char *> keys, vals;
            keys.reserve(options.size());
            vals.reserve(options.size());
            for (const auto &kv : options)
            {
                keys.push_back(kv.first.c_str());
                vals.push_back(kv.second.c_str());
            }
            static const std::unordered_map<std::string, std::string> kAlias = {
                {"CPU", "CPUExecutionProvider"},
                {"CUDA", "CUDAExecutionProvider"},
                {"DML", "DmlExecutionProvider"},
                {"DirectML", "DmlExecutionProvider"},
                {"CoreML", "CoreMLExecutionProvider"},
                {"QNN", "QNNExecutionProvider"},
                {"TensorRT", "TensorrtExecutionProvider"}};
            std::string canonical = name;
            auto it = kAlias.find(name);
            if (it != kAlias.end())
                canonical = it->second;

            if (canonical == "CPUExecutionProvider")
                return;

#ifdef ORTB_HAS_DML
            if (canonical == "DmlExecutionProvider")
            {
                int device_id = 0;
                auto it = options.find("device_id");
                if (it != options.end())
                {
                    try
                    {
                        device_id = std::stoi(it->second);
                    }
                    catch (...)
                    {
                    }
                }
                opts.DisableMemPattern();
                opts.SetExecutionMode(ORT_SEQUENTIAL);
                Ort::ThrowOnError(
                    OrtSessionOptionsAppendExecutionProvider_DML(opts, device_id));
                return;
            }
#endif

            opts.AppendExecutionProvider(canonical.c_str(), options);
        }

    } // namespace

    void apply(const bnl_api *api, bnl_value *map, Ort::SessionOptions &out)
    {
        if (!map || api->get_type(map) != BNL_TYPE_MAP)
            return;

        if (api->map_has(map, "log_severity_level"))
        {
            int lvl = static_cast<int>(get_number_or_throw(api, map, "log_severity_level"));
            out.SetLogSeverityLevel(lvl);
        }
        if (api->map_has(map, "intra_op_num_threads"))
        {
            int n = static_cast<int>(get_number_or_throw(api, map, "intra_op_num_threads"));
            out.SetIntraOpNumThreads(n);
        }
        if (api->map_has(map, "inter_op_num_threads"))
        {
            int n = static_cast<int>(get_number_or_throw(api, map, "inter_op_num_threads"));
            out.SetInterOpNumThreads(n);
        }

        if (api->map_has(map, "execution_providers"))
        {
            bnl_value *list = api->map_get(api, map, "execution_providers");
            if (api->get_type(list) != BNL_TYPE_LIST)
                throw std::runtime_error("session option 'execution_providers' must be a list");
            std::size_t n = api->list_length(list);
            for (std::size_t i = 0; i < n; ++i)
            {
                bnl_value *entry = api->list_get(api, list, i);
                int t = api->get_type(entry);
                if (t == BNL_TYPE_STRING)
                {
                    std::size_t slen = 0;
                    const char *sp = api->get_string(entry, &slen);
                    append_provider(std::string(sp, slen), {}, out);
                }
                else if (t == BNL_TYPE_MAP)
                {
                    if (!api->map_has(entry, "name"))
                        throw std::runtime_error("execution_providers entry: missing 'name'");
                    bnl_value *nv = api->map_get(api, entry, "name");
                    std::size_t nlen = 0;
                    const char *np = api->get_string(nv, &nlen);
                    std::string ep_name(np, nlen);
                    std::unordered_map<std::string, std::string> ep_opts;
                    std::size_t msz = api->map_size(entry);
                    for (std::size_t j = 0; j < msz; ++j)
                    {
                        const char *k = nullptr;
                        std::size_t klen = 0;
                        api->map_key_at(entry, j, &k, &klen);
                        std::string key(k, klen);
                        if (key == "name")
                            continue;
                        bnl_value *val = api->map_get(api, entry, key.c_str());
                        if (api->get_type(val) == BNL_TYPE_STRING)
                        {
                            std::size_t vlen = 0;
                            const char *vp = api->get_string(val, &vlen);
                            ep_opts[key] = std::string(vp, vlen);
                        }
                        else if (api->get_type(val) == BNL_TYPE_NUMBER)
                        {
                            ep_opts[key] = std::to_string(static_cast<long long>(api->get_number(val)));
                        }
                    }
                    append_provider(ep_name, ep_opts, out);
                }
                else
                {
                    throw std::runtime_error("execution_providers entry must be a string or map");
                }
            }
        }
    }
} // namespace ortb::session_options
