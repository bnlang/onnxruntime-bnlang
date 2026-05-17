#include "session.h"
#include "tensor.h"
#include "session_options.h"

#include <onnxruntime_cxx_api.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace ortb::session
{

    namespace
    {

        Ort::Env &env()
        {
            static Ort::Env e(ORT_LOGGING_LEVEL_WARNING, "onnxruntime-bnlang");
            return e;
        }

        struct Table
        {
            std::mutex mu;
            int next_id = 1;
            std::unordered_map<int, std::unique_ptr<Ort::Session>> sessions;
        };

        Table &table()
        {
            static Table t;
            return t;
        }

#ifdef _WIN32
        std::wstring to_native_path(const char *s, std::size_t len)
        {
            if (len == 0)
                return {};
            int n = MultiByteToWideChar(CP_UTF8, 0, s, static_cast<int>(len), nullptr, 0);
            if (n <= 0)
                return {};
            std::wstring out(static_cast<std::size_t>(n), L'\0');
            MultiByteToWideChar(CP_UTF8, 0, s, static_cast<int>(len), out.data(), n);
            return out;
        }
#else
        std::string to_native_path(const char *s, std::size_t len)
        {
            return std::string(s, len);
        }
#endif

        // Build a {name, dtype, shape} map for a single input/output slot.
        bnl_value *build_slot_info(const bnl_api *api,
                                   const char *name,
                                   const Ort::TypeInfo &type_info)
        {
            bnl_value *item = api->make_map(api);
            api->map_set(item, "name",
                         api->make_string(api, name, std::strlen(name)));

            if (type_info.GetONNXType() == ONNX_TYPE_TENSOR)
            {
                auto tsi = type_info.GetTensorTypeAndShapeInfo();
                const char *dtype = ortb::tensor::dtype_to_string(tsi.GetElementType());
                api->map_set(item, "dtype",
                             api->make_string(api, dtype, std::strlen(dtype)));

                bnl_value *shape = api->make_list(api);
                for (int64_t d : tsi.GetShape())
                {
                    api->list_push(shape, api->make_number(api, static_cast<double>(d)));
                }
                api->map_set(item, "shape", shape);
            }
            else
            {
                api->map_set(item, "dtype",
                             api->make_string(api, "non-tensor", 10));
            }
            return item;
        }

        // ----------------- session_open ----------------------------------------
        bnl_value *session_open_fn(const bnl_api *api, int argc, bnl_value **argv, void *ud)
        {
            (void)argc;
            (void)ud;
            try
            {
                std::size_t plen = 0;
                const char *pbytes = api->get_string(argv[0], &plen);
                auto path = to_native_path(pbytes, plen);

                Ort::SessionOptions opts;
                ortb::session_options::apply(api, argv[1], opts);

                auto sess = std::make_unique<Ort::Session>(env(), path.c_str(), opts);

                Table &t = table();
                std::lock_guard<std::mutex> lock(t.mu);
                int id = t.next_id++;
                t.sessions.emplace(id, std::move(sess));
                return api->make_number(api, static_cast<double>(id));
            }
            catch (const std::exception &e)
            {
                api->throw_error(api, e.what());
                return nullptr;
            }
            catch (...)
            {
                api->throw_error(api, "session_open: unknown error");
                return nullptr;
            }
        }

        // ----------------- session_close ---------------------------------------
        bnl_value *session_close_fn(const bnl_api *api, int argc, bnl_value **argv, void *ud)
        {
            (void)argc;
            (void)ud;
            try
            {
                int id = static_cast<int>(api->get_number(argv[0]));
                Table &t = table();
                std::lock_guard<std::mutex> lock(t.mu);
                t.sessions.erase(id);
                return api->make_null(api);
            }
            catch (const std::exception &e)
            {
                api->throw_error(api, e.what());
                return nullptr;
            }
        }

        // ----------------- session_info ----------------------------------------
        bnl_value *session_info_fn(const bnl_api *api, int argc, bnl_value **argv, void *ud)
        {
            (void)argc;
            (void)ud;
            try
            {
                int id = static_cast<int>(api->get_number(argv[0]));

                Ort::Session *sess = nullptr;
                {
                    Table &t = table();
                    std::lock_guard<std::mutex> lock(t.mu);
                    auto it = t.sessions.find(id);
                    if (it == t.sessions.end())
                        throw std::runtime_error("invalid session handle");
                    sess = it->second.get();
                }

                Ort::AllocatorWithDefaultOptions alloc;
                bnl_value *result = api->make_map(api);

                bnl_value *inputs = api->make_list(api);
                for (std::size_t i = 0, n = sess->GetInputCount(); i < n; ++i)
                {
                    auto namep = sess->GetInputNameAllocated(i, alloc);
                    auto info = sess->GetInputTypeInfo(i);
                    api->list_push(inputs, build_slot_info(api, namep.get(), info));
                }
                api->map_set(result, "inputs", inputs);

                bnl_value *outputs = api->make_list(api);
                for (std::size_t i = 0, n = sess->GetOutputCount(); i < n; ++i)
                {
                    auto namep = sess->GetOutputNameAllocated(i, alloc);
                    auto info = sess->GetOutputTypeInfo(i);
                    api->list_push(outputs, build_slot_info(api, namep.get(), info));
                }
                api->map_set(result, "outputs", outputs);

                return result;
            }
            catch (const std::exception &e)
            {
                api->throw_error(api, e.what());
                return nullptr;
            }
            catch (...)
            {
                api->throw_error(api, "session_info: unknown error");
                return nullptr;
            }
        }

        // ----------------- session_run -----------------------------------------
        bnl_value *session_run_fn(const bnl_api *api, int argc, bnl_value **argv, void *ud)
        {
            (void)argc;
            (void)ud;
            try
            {
                int id = static_cast<int>(api->get_number(argv[0]));
                bnl_value *feeds = argv[1];

                if (api->get_type(feeds) != BNL_TYPE_MAP)
                    throw std::runtime_error("session_run: feeds must be a map");

                Ort::Session *sess = nullptr;
                {
                    Table &t = table();
                    std::lock_guard<std::mutex> lock(t.mu);
                    auto it = t.sessions.find(id);
                    if (it == t.sessions.end())
                        throw std::runtime_error("invalid session handle");
                    sess = it->second.get();
                }

                std::size_t in_count = api->map_size(feeds);
                std::vector<std::unique_ptr<ortb::tensor::InputBuffer>> buffers;
                buffers.reserve(in_count);
                std::vector<std::string> in_name_storage; // owns C-string memory
                std::vector<const char *> in_names;
                std::vector<Ort::Value> in_values;
                in_name_storage.reserve(in_count);
                in_names.reserve(in_count);
                in_values.reserve(in_count);

                for (std::size_t i = 0; i < in_count; ++i)
                {
                    const char *k = nullptr;
                    std::size_t klen = 0;
                    api->map_key_at(feeds, i, &k, &klen);
                    std::string name(k, klen);

                    bnl_value *spec = api->map_get(api, feeds, name.c_str());
                    auto buf = ortb::tensor::from_bnl(api, spec, name);

                    in_name_storage.push_back(std::move(name));
                    in_names.push_back(in_name_storage.back().c_str());
                    in_values.push_back(std::move(buf->value));
                    buffers.push_back(std::move(buf));
                }

                Ort::AllocatorWithDefaultOptions alloc;
                std::size_t out_count = sess->GetOutputCount();
                std::vector<Ort::AllocatedStringPtr> out_name_storage;
                std::vector<const char *> out_names;
                out_name_storage.reserve(out_count);
                out_names.reserve(out_count);
                for (std::size_t i = 0; i < out_count; ++i)
                {
                    out_name_storage.push_back(sess->GetOutputNameAllocated(i, alloc));
                    out_names.push_back(out_name_storage.back().get());
                }

                Ort::RunOptions run_opts;
                auto out_values = sess->Run(
                    run_opts,
                    in_names.data(), in_values.data(), in_count,
                    out_names.data(), out_count);

                bnl_value *result = api->make_map(api);
                for (std::size_t i = 0; i < out_count; ++i)
                {
                    bnl_value *item = ortb::tensor::to_bnl_handle(api, std::move(out_values[i]));
                    api->map_set(result, out_names[i], item);
                }
                return result;
            }
            catch (const std::exception &e)
            {
                api->throw_error(api, e.what());
                return nullptr;
            }
            catch (...)
            {
                api->throw_error(api, "session_run: unknown error");
                return nullptr;
            }
        }

    } // namespace

    void register_natives(const bnl_api *api, bnl_module *mod)
    {
        (void)api;
        api->module_add_function(mod, "session_open", 2, &session_open_fn, nullptr);
        api->module_add_function(mod, "session_close", 1, &session_close_fn, nullptr);
        api->module_add_function(mod, "session_info", 1, &session_info_fn, nullptr);
        api->module_add_function(mod, "session_run", 2, &session_run_fn, nullptr);
    }

} // namespace ortb::session
