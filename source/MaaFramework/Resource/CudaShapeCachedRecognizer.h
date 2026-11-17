#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <list>
#include <memory>
#include <mutex>
#include <string_view>
#include <utility>
#include <vector>

#ifdef _WIN32
#include "MaaUtils/SafeWindows.hpp"
#endif

#include "fastdeploy/vision/ocr/ppocr/recognizer.h"

namespace maa_ocr
{
class CudaShapeCachedRecognizer final : public fastdeploy::vision::ocr::Recognizer
{
public:
    using Recognizer::Recognizer;

    bool Infer(std::vector<fastdeploy::FDTensor>& input, std::vector<fastdeploy::FDTensor>* output) override
    {
        if (!m_cache_enabled || input.empty() || runtime_option.device != fastdeploy::Device::CUDA
            || runtime_option.backend != fastdeploy::Backend::ORT || EnabledRecordTimeOfRuntime()) {
            return Recognizer::Infer(input, output);
        }

        std::vector<TensorKey> key;
        key.reserve(input.size());
        for (const auto& tensor : input) {
            key.push_back({ tensor.shape, tensor.dtype });
        }

        std::lock_guard guard(m_mutex);
        const auto found = std::ranges::find(m_sessions, key, &Entry::key);
        if (found != m_sessions.end()) {
            m_sessions.splice(m_sessions.begin(), m_sessions, found);
        }
        else {
            if (m_sessions.size() == Capacity) {
                m_sessions.pop_back();
            }
            // FastDeploy's ORT clone creates an independent session with the same runtime options.
            std::unique_ptr<fastdeploy::Runtime> session(CloneRuntime());
            if (!session) {
                return Recognizer::Infer(input, output);
            }
            m_sessions.push_front({ std::move(key), std::move(session) });
        }
        return m_sessions.front().runtime->Infer(input, output);
    }

private:
    static bool cache_enabled()
    {
#ifdef _WIN32
        char value[2] { };
        return GetEnvironmentVariableA("MAA_OCR_SHAPE_CACHE", value, sizeof(value)) == 1 && value[0] == '1';
#else
        const auto* value = std::getenv("MAA_OCR_SHAPE_CACHE");
        return value && std::string_view(value) == "1";
#endif
    }

    struct TensorKey
    {
        std::vector<std::int64_t> shape;
        fastdeploy::FDDataType dtype;
        bool operator==(const TensorKey&) const = default;
    };

    struct Entry
    {
        std::vector<TensorKey> key;
        std::unique_ptr<fastdeploy::Runtime> runtime;
    };

    static constexpr std::size_t Capacity = 4;
    const bool m_cache_enabled = cache_enabled();
    std::mutex m_mutex;
    std::list<Entry> m_sessions;
};
}
