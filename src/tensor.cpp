#include "tensor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ortb::tensor
{
    namespace
    {

        struct TensorTable
        {
            std::mutex mu;
            int next_id = 1;
            std::unordered_map<int, Ort::Value> values;
        };

        TensorTable &table()
        {
            static TensorTable t;
            return t;
        }

        int store_value(Ort::Value &&v)
        {
            auto &t = table();
            std::lock_guard<std::mutex> lk(t.mu);
            int id = t.next_id++;
            t.values.emplace(id, std::move(v));
            return id;
        }

        Ort::Value take_value(int id)
        {
            auto &t = table();
            std::lock_guard<std::mutex> lk(t.mu);
            auto it = t.values.find(id);
            if (it == t.values.end())
                throw std::runtime_error("tensor: invalid or already-consumed handle " + std::to_string(id));
            Ort::Value out = std::move(it->second);
            t.values.erase(it);
            return out;
        }

        const Ort::Value *peek_value(int id)
        {
            auto &t = table();
            std::lock_guard<std::mutex> lk(t.mu);
            auto it = t.values.find(id);
            return it == t.values.end() ? nullptr : &it->second;
        }

    } // namespace

    const char *dtype_to_string(ONNXTensorElementDataType t)
    {
        switch (t)
        {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return "float64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            return "int16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return "int8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return "bool";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return "float16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            return "string";
        default:
            return "unsupported";
        }
    }

    namespace
    {
        const Ort::MemoryInfo &cpu_mem()
        {
            static auto mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            return mi;
        }

        std::string get_string_field(const bnl_api *api, bnl_value *m,
                                     const char *key, const std::string &ctx)
        {
            if (!api->map_has(m, key))
                throw std::runtime_error(ctx + ": missing '" + key + "'");
            bnl_value *v = api->map_get(api, m, key);
            if (api->get_type(v) != BNL_TYPE_STRING)
                throw std::runtime_error(ctx + ": '" + std::string(key) + "' must be a string");
            std::size_t n = 0;
            const char *p = api->get_string(v, &n);
            return std::string(p, n);
        }

        std::vector<int64_t> get_shape_field(const bnl_api *api, bnl_value *m,
                                             const std::string &ctx)
        {
            if (!api->map_has(m, "shape"))
                throw std::runtime_error(ctx + ": missing 'shape'");
            bnl_value *sh = api->map_get(api, m, "shape");
            if (api->get_type(sh) != BNL_TYPE_LIST)
                throw std::runtime_error(ctx + ": 'shape' must be a list");
            std::vector<int64_t> out;
            std::size_t n = api->list_length(sh);
            out.reserve(n);
            for (std::size_t i = 0; i < n; ++i)
            {
                bnl_value *dim = api->list_get(api, sh, i);
                if (api->get_type(dim) != BNL_TYPE_NUMBER)
                    throw std::runtime_error(ctx + ": shape dims must be numbers");
                out.push_back(static_cast<int64_t>(api->get_number(dim)));
            }
            return out;
        }

        bnl_value *get_data_field(const bnl_api *api, bnl_value *m,
                                  const std::string &ctx)
        {
            if (!api->map_has(m, "data"))
                throw std::runtime_error(ctx + ": missing 'data'");
            bnl_value *d = api->map_get(api, m, "data");
            if (api->get_type(d) != BNL_TYPE_LIST)
                throw std::runtime_error(ctx + ": 'data' must be a list");
            return d;
        }

        std::size_t elem_count(const std::vector<int64_t> &shape)
        {
            std::size_t n = 1;
            for (auto d : shape)
            {
                if (d < 0)
                    throw std::runtime_error("negative dim in input shape");
                n *= static_cast<std::size_t>(d);
            }
            return n;
        }

    } // namespace

    std::unique_ptr<InputBuffer> from_bnl(const bnl_api *api,
                                          bnl_value *spec,
                                          const std::string &input_name)
    {
        const std::string ctx = "input '" + input_name + "'";
        if (api->get_type(spec) != BNL_TYPE_MAP)
            throw std::runtime_error(ctx + ": must be a map { dtype, shape, data | handle }");

        if (api->map_has(spec, "handle"))
        {
            bnl_value *hv = api->map_get(api, spec, "handle");
            if (api->get_type(hv) != BNL_TYPE_NUMBER)
                throw std::runtime_error(ctx + ": 'handle' must be a number");
            int id = static_cast<int>(api->get_number(hv));
            auto buf = std::make_unique<InputBuffer>();
            buf->value = take_value(id);
            return buf;
        }

        std::string dtype = get_string_field(api, spec, "dtype", ctx);
        std::vector<int64_t> shape = get_shape_field(api, spec, ctx);
        bnl_value *data_list = get_data_field(api, spec, ctx);

        std::size_t expected = elem_count(shape);
        std::size_t given = api->list_length(data_list);
        if (given != expected)
        {
            throw std::runtime_error(ctx + ": data has " + std::to_string(given) +
                                     " items but shape expects " + std::to_string(expected));
        }

        auto buf = std::make_unique<InputBuffer>();
        buf->shape = shape;

        if (dtype == "float32")
        {
            buf->f32.resize(expected);
            for (std::size_t i = 0; i < expected; ++i)
            {
                bnl_value *v = api->list_get(api, data_list, i);
                buf->f32[i] = static_cast<float>(api->get_number(v));
            }
            buf->value = Ort::Value::CreateTensor<float>(
                cpu_mem(), buf->f32.data(), buf->f32.size(),
                buf->shape.data(), buf->shape.size());
        }
        else if (dtype == "float64")
        {
            buf->f64.resize(expected);
            for (std::size_t i = 0; i < expected; ++i)
            {
                bnl_value *v = api->list_get(api, data_list, i);
                buf->f64[i] = api->get_number(v);
            }
            buf->value = Ort::Value::CreateTensor<double>(
                cpu_mem(), buf->f64.data(), buf->f64.size(),
                buf->shape.data(), buf->shape.size());
        }
        else if (dtype == "float16")
        {
            buf->f16.reserve(expected);
            for (std::size_t i = 0; i < expected; ++i)
            {
                bnl_value *v = api->list_get(api, data_list, i);
                buf->f16.emplace_back(static_cast<float>(api->get_number(v)));
            }
            buf->value = Ort::Value::CreateTensor<Ort::Float16_t>(
                cpu_mem(), buf->f16.data(), buf->f16.size(),
                buf->shape.data(), buf->shape.size());
        }
        else if (dtype == "int64")
        {
            buf->i64.resize(expected);
            for (std::size_t i = 0; i < expected; ++i)
            {
                bnl_value *v = api->list_get(api, data_list, i);
                buf->i64[i] = static_cast<int64_t>(api->get_number(v));
            }
            buf->value = Ort::Value::CreateTensor<int64_t>(
                cpu_mem(), buf->i64.data(), buf->i64.size(),
                buf->shape.data(), buf->shape.size());
        }
        else if (dtype == "int32")
        {
            buf->i32.resize(expected);
            for (std::size_t i = 0; i < expected; ++i)
            {
                bnl_value *v = api->list_get(api, data_list, i);
                buf->i32[i] = static_cast<int32_t>(api->get_number(v));
            }
            buf->value = Ort::Value::CreateTensor<int32_t>(
                cpu_mem(), buf->i32.data(), buf->i32.size(),
                buf->shape.data(), buf->shape.size());
        }
        else if (dtype == "uint8")
        {
            buf->u8.resize(expected);
            for (std::size_t i = 0; i < expected; ++i)
            {
                bnl_value *v = api->list_get(api, data_list, i);
                buf->u8[i] = static_cast<uint8_t>(api->get_number(v));
            }
            buf->value = Ort::Value::CreateTensor<uint8_t>(
                cpu_mem(), buf->u8.data(), buf->u8.size(),
                buf->shape.data(), buf->shape.size());
        }
        else if (dtype == "bool")
        {
            buf->b8.resize(expected);
            for (std::size_t i = 0; i < expected; ++i)
            {
                bnl_value *v = api->list_get(api, data_list, i);
                // Accept either a bnl bool OR a number (1/0) — scripts commonly
                // pass attention masks as 0/1 integers.
                int t = api->get_type(v);
                buf->b8[i] = (t == BNL_TYPE_BOOL)     ? (api->get_bool(v) ? 1u : 0u)
                             : (t == BNL_TYPE_NUMBER) ? (api->get_number(v) != 0.0 ? 1u : 0u)
                                                      : throw std::runtime_error(ctx + ": bool element must be bool or number");
            }
            buf->value = Ort::Value::CreateTensor(
                cpu_mem(), buf->b8.data(), buf->b8.size() * sizeof(uint8_t),
                buf->shape.data(), buf->shape.size(),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
        }
        else
        {
            throw std::runtime_error(ctx + ": unsupported dtype '" + dtype +
                                     "' (supported: float32, float64, float16, int64, int32, uint8, bool)");
        }

        return buf;
    }

    namespace
    {

        void set_dtype_and_shape(const bnl_api *api, bnl_value *out, const Ort::Value &v)
        {
            auto info = v.GetTensorTypeAndShapeInfo();
            auto type = info.GetElementType();
            auto shape = info.GetShape();

            const char *dtname = dtype_to_string(type);
            api->map_set(out, "dtype",
                         api->make_string(api, dtname, std::strlen(dtname)));

            bnl_value *sh = api->make_list(api);
            for (int64_t d : shape)
            {
                api->list_push(sh, api->make_number(api, static_cast<double>(d)));
            }
            api->map_set(out, "shape", sh);
        }

        bnl_value *materialize_data(const bnl_api *api, const Ort::Value &v)
        {
            auto info = v.GetTensorTypeAndShapeInfo();
            auto type = info.GetElementType();
            std::size_t count = info.GetElementCount();

            bnl_value *data = api->make_list(api);
            switch (type)
            {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            {
                const float *p = v.GetTensorData<float>();
                for (std::size_t i = 0; i < count; ++i)
                    api->list_push(data, api->make_number(api, static_cast<double>(p[i])));
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            {
                const double *p = v.GetTensorData<double>();
                for (std::size_t i = 0; i < count; ++i)
                    api->list_push(data, api->make_number(api, p[i]));
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            {
                const Ort::Float16_t *p = v.GetTensorData<Ort::Float16_t>();
                for (std::size_t i = 0; i < count; ++i)
                    api->list_push(data, api->make_number(api, static_cast<double>(p[i].ToFloat())));
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            {
                const int64_t *p = v.GetTensorData<int64_t>();
                for (std::size_t i = 0; i < count; ++i)
                    api->list_push(data, api->make_number(api, static_cast<double>(p[i])));
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            {
                const int32_t *p = v.GetTensorData<int32_t>();
                for (std::size_t i = 0; i < count; ++i)
                    api->list_push(data, api->make_number(api, static_cast<double>(p[i])));
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            {
                const uint8_t *p = v.GetTensorData<uint8_t>();
                for (std::size_t i = 0; i < count; ++i)
                    api->list_push(data, api->make_number(api, static_cast<double>(p[i])));
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            {
                const bool *p = v.GetTensorData<bool>();
                for (std::size_t i = 0; i < count; ++i)
                    api->list_push(data, api->make_bool(api, p[i] ? 1 : 0));
                break;
            }
            default:
                break;
            }
            return data;
        }

    } // namespace

    bnl_value *to_bnl_handle(const bnl_api *api, Ort::Value &&v)
    {
        bnl_value *out = api->make_map(api);
        set_dtype_and_shape(api, out, v);
        int handle = store_value(std::move(v));
        api->map_set(out, "handle", api->make_number(api, static_cast<double>(handle)));
        return out;
    }

    bnl_value *tensor_to_data(const bnl_api *api, int handle)
    {
        const Ort::Value *v = peek_value(handle);
        if (!v)
            throw std::runtime_error("tensor_to_data: invalid handle " + std::to_string(handle));
        return materialize_data(api, *v);
    }

    void tensor_close(int handle)
    {
        auto &t = table();
        std::lock_guard<std::mutex> lk(t.mu);
        t.values.erase(handle);
    }

    // --- Handle-based sampling helpers ---------------------------------------
    namespace
    {
        std::pair<const float *, std::size_t> peek_float_buffer(int handle)
        {
            auto &t = table();
            std::lock_guard<std::mutex> lk(t.mu);
            auto it = t.values.find(handle);
            if (it == t.values.end())
                throw std::runtime_error("invalid tensor handle " +
                                         std::to_string(handle));
            const Ort::Value &v = it->second;
            auto info = v.GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                throw std::runtime_error("tensor handle is not float32 logits");
            return { v.GetTensorData<float>(), info.GetElementCount() };
        }
    } // namespace

    int tensor_argmax_last(int handle, int vocab)
    {
        if (vocab <= 0)
            throw std::runtime_error("argmax_last: vocab must be positive");
        auto buf = peek_float_buffer(handle);
        if (buf.second < static_cast<std::size_t>(vocab))
            throw std::runtime_error("argmax_last: tensor shorter than vocab");

        const float *p = buf.first + (buf.second - static_cast<std::size_t>(vocab));
        float best = p[0];
        int best_i = 0;
        for (int i = 1; i < vocab; ++i)
        {
            if (p[i] > best) { best = p[i]; best_i = i; }
        }
        return best_i;
    }

    int tensor_sample_last(int handle, int vocab,
                           double temperature, int top_k,
                           double top_p, double seed_val)
    {
        if (vocab <= 0)
            throw std::runtime_error("sample_last: vocab must be positive");
        auto buf = peek_float_buffer(handle);
        if (buf.second < static_cast<std::size_t>(vocab))
            throw std::runtime_error("sample_last: tensor shorter than vocab");

        const float *src = buf.first + (buf.second - static_cast<std::size_t>(vocab));

        // temperature<=0 ⇒ argmax (greedy fast path).
        if (temperature <= 0.0)
        {
            float best = src[0];
            int best_i = 0;
            for (int i = 1; i < vocab; ++i)
                if (src[i] > best) { best = src[i]; best_i = i; }
            return best_i;
        }

        // Effective k: vocab when top_k<=0 or top_k>=vocab.
        const int k = (top_k > 0 && top_k < vocab) ? top_k : vocab;

        std::vector<std::pair<float, int>> heap;
        heap.reserve(static_cast<std::size_t>(k));
        auto cmp = [](const std::pair<float, int> &a,
                      const std::pair<float, int> &b)
        { return a.first > b.first; };

        // Seed heap with first k elements.
        for (int i = 0; i < k; ++i)
            heap.emplace_back(src[i], i);
        std::make_heap(heap.begin(), heap.end(), cmp);

        // Walk the rest, pushing into heap only when better than current min.
        for (int i = k; i < vocab; ++i)
        {
            if (src[i] > heap.front().first)
            {
                std::pop_heap(heap.begin(), heap.end(), cmp);
                heap.back() = { src[i], i };
                std::push_heap(heap.begin(), heap.end(), cmp);
            }
        }

        // Apply temperature now (only k values, not vocab).
        const float inv_t = static_cast<float>(1.0 / temperature);
        for (auto &p : heap) p.first *= inv_t;

        // Softmax over the k retained logits.
        float max_l = heap[0].first;
        for (const auto &p : heap)
            if (p.first > max_l) max_l = p.first;

        std::vector<double> probs(heap.size());
        double sum = 0.0;
        for (std::size_t i = 0; i < heap.size(); ++i)
        {
            double p = std::exp(static_cast<double>(heap[i].first - max_l));
            probs[i] = p;
            sum += p;
        }
        for (double &p : probs) p /= sum;

        // Optional top-p nucleus filter (cheap: at most k elements).
        if (top_p > 0.0 && top_p < 1.0)
        {
            std::vector<std::size_t> idx(probs.size());
            for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
            std::sort(idx.begin(), idx.end(),
                      [&](std::size_t a, std::size_t b)
                      { return probs[a] > probs[b]; });
            double cum = 0.0;
            std::size_t keep = 0;
            for (; keep < idx.size(); ++keep)
            {
                cum += probs[idx[keep]];
                if (cum >= top_p) { ++keep; break; }
            }
            std::vector<std::pair<float, int>> filt_l;
            std::vector<double> filt_p;
            filt_l.reserve(keep);
            filt_p.reserve(keep);
            double s = 0.0;
            for (std::size_t kk = 0; kk < keep; ++kk)
            {
                filt_l.push_back(heap[idx[kk]]);
                filt_p.push_back(probs[idx[kk]]);
                s += probs[idx[kk]];
            }
            for (double &p : filt_p) p /= s;
            heap.swap(filt_l);
            probs.swap(filt_p);
        }

        static std::mt19937_64 rng{ std::random_device{}() };
        if (seed_val != 0.0)
            rng.seed(static_cast<std::uint64_t>(seed_val));
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double u = dist(rng);

        double cum = 0.0;
        int pick = heap.back().second;
        for (std::size_t i = 0; i < probs.size(); ++i)
        {
            cum += probs[i];
            if (u < cum) { pick = heap[i].second; break; }
        }
        return pick;
    }

    // ----------------- native functions -------------------------------------
    namespace
    {

        bnl_value *tensor_to_data_fn(const bnl_api *api, int argc, bnl_value **argv, void *ud)
        {
            (void)argc;
            (void)ud;
            try
            {
                int h = static_cast<int>(api->get_number(argv[0]));
                return tensor_to_data(api, h);
            }
            catch (const std::exception &e)
            {
                api->throw_error(api, e.what());
                return nullptr;
            }
        }

        bnl_value *tensor_close_fn(const bnl_api *api, int argc, bnl_value **argv, void *ud)
        {
            (void)argc;
            (void)ud;
            try
            {
                int h = static_cast<int>(api->get_number(argv[0]));
                tensor_close(h);
                return api->make_null(api);
            }
            catch (const std::exception &e)
            {
                api->throw_error(api, e.what());
                return nullptr;
            }
        }

        bnl_value *tensor_argmax_last_fn(const bnl_api *api, int argc, bnl_value **argv, void *ud)
        {
            (void)argc;
            (void)ud;
            try
            {
                int h = static_cast<int>(api->get_number(argv[0]));
                int v = static_cast<int>(api->get_number(argv[1]));
                return api->make_number(api, static_cast<double>(tensor_argmax_last(h, v)));
            }
            catch (const std::exception &e)
            {
                api->throw_error(api, e.what());
                return nullptr;
            }
        }

        bnl_value *tensor_sample_last_fn(const bnl_api *api, int argc, bnl_value **argv, void *ud)
        {
            (void)argc;
            (void)ud;
            try
            {
                int h = static_cast<int>(api->get_number(argv[0]));
                int v = static_cast<int>(api->get_number(argv[1]));
                double temp = api->get_number(argv[2]);
                int top_k = static_cast<int>(api->get_number(argv[3]));
                double top_p = api->get_number(argv[4]);
                double seed_val = api->get_number(argv[5]);
                int picked = tensor_sample_last(h, v, temp, top_k, top_p, seed_val);
                return api->make_number(api, static_cast<double>(picked));
            }
            catch (const std::exception &e)
            {
                api->throw_error(api, e.what());
                return nullptr;
            }
        }

    } // namespace

    void register_tensor_natives(const bnl_api *api, bnl_module *mod)
    {
        (void)api;
        api->module_add_function(mod, "tensor_to_data", 1, &tensor_to_data_fn, nullptr);
        api->module_add_function(mod, "tensor_close", 1, &tensor_close_fn, nullptr);
        api->module_add_function(mod, "tensor_argmax_last", 2, &tensor_argmax_last_fn, nullptr);
        api->module_add_function(mod, "tensor_sample_last", 6, &tensor_sample_last_fn, nullptr);
    }

} // namespace ortb::tensor
