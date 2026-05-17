#include "bnl/plugin.h"
#include "session.h"
#include "tensor.h"

#include <onnxruntime_cxx_api.h>

#include <string>

#define ORT_BNL_VERSION "1.0.0"

namespace
{

    bnl_value *providers_fn(const bnl_api *api, int argc, bnl_value **argv, void *ud)
    {
        (void)argc;
        (void)argv;
        (void)ud;
        bnl_value *list = api->make_list(api);
        try
        {
            for (const auto &p : Ort::GetAvailableProviders())
            {
                api->list_push(list, api->make_string(api, p.data(), p.size()));
            }
        }
        catch (const std::exception &e)
        {
            api->throw_error(api, e.what());
            return nullptr;
        }
        catch (...)
        {
            api->throw_error(api, "providers: unknown error");
            return nullptr;
        }
        return list;
    }

} // namespace

extern "C" BNL_EXPORT bnl_module *bnl_load(const bnl_api *api)
{
    bnl_module *mod = api->module_new(api, "onnxruntime-bnlang");

    std::string ver = ORT_BNL_VERSION " (ort ";
    ver += Ort::GetVersionString();
    ver += ")";
    api->module_add_value(mod, "version",
                          api->make_string(api, ver.data(), ver.size()));

    api->module_add_function(mod, "providers", 0, &providers_fn, nullptr);

    ortb::session::register_natives(api, mod);
    ortb::tensor::register_tensor_natives(api, mod);

    return mod;
}
