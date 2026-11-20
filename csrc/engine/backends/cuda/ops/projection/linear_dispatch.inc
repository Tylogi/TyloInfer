template <typename Decode>
static ::tyloi::DenseTensor sq_matmul_cpu_impl(
        ::tyloi::DenseTensor input,
        int64_t outputs,
        int64_t width,
        Decode decode) {
    if (!(!input.is_cuda() && input.size(-1) == width))
        throw std::runtime_error("SQ CPU activation device or width mismatch");
    const auto dtype = input.scalar_type();
    auto shape = input.sizes().vec();
    input = input.to(::tyloi::kFloat32)
        .reshape({-1, width}).contiguous();
    const auto rows = input.size(0);
    auto result = ::tyloi::cuda::empty({rows, outputs}, input.options());
    const auto* source = input.data_ptr<float>();
    auto* destination = result.data_ptr<float>();
    mfq_parallel_for(0, outputs, 1, [&](int64_t begin, int64_t end) {
        std::vector<float> weights(width);
        for (auto output = begin; output < end; ++output) {
            decode(output, weights.data());
            for (int64_t row = 0; row < rows; ++row) {
                float sum = 0;
                for (int64_t column = 0; column < width; ++column) {
                    sum = std::fma(
                        source[row * width + column], weights[column], sum);
                }
                destination[row * outputs + output] = sum;
            }
        }
    });
    shape.back() = outputs;
    return result.to(dtype).reshape(shape);
}

::tyloi::DenseTensor sq_matmul_cpu(
        const tyloi::Mxfp4SqTensor& weight,
        ::tyloi::DenseTensor input) {
    if (weight.blob.is_cuda())
        throw std::runtime_error("SQ CPU weights must be on CPU");
    const auto* blob = weight.blob.data_ptr<uint8_t>();
    const auto layout = tyloi::sq::parse(blob, weight.blob.numel());
    return sq_matmul_cpu_impl(
        std::move(input), weight.out, weight.neuron_len,
        [&](int64_t row, float* output) {
            tyloi::sq::decode_cpu_row(
                blob, layout, weight.row_q.data_ptr<uint8_t>()[row],
                weight.row_symbol_byte_offsets.data_ptr<int32_t>()[row],
                weight.row_auxiliary.data_ptr<int32_t>()[row], output);
        });
}

::tyloi::DenseTensor sq_matmul_cpu(
        const tyloi::Fp8SqTensor& weight,
        ::tyloi::DenseTensor input) {
    if (weight.blob.is_cuda())
        throw std::runtime_error("SQ CPU weights must be on CPU");
    const auto* blob = weight.blob.data_ptr<uint8_t>();
    const auto layout = tyloi::fp8sq::parse(
        weight.dtype, blob, weight.blob.numel());
    return sq_matmul_cpu_impl(
        std::move(input), weight.out, weight.neuron_len,
        [&](int64_t row, float* output) {
            for (int64_t column = 0; column < weight.neuron_len; ++column) {
                output[column] = tyloi::fp8sq::decode_cpu(
                    blob, layout, weight.row_q.data_ptr<uint8_t>()[row],
                    weight.row_symbol_byte_offsets.data_ptr<int32_t>()[row],
                    row, column);
            }
        });
}

template<class Weight>
static tyloi::DenseTensor scaled_prefill(
        const Weight& weight, const tyloi::DenseTensor& input,
        tyloi::cuda::gguf::GgufPrefillProduct* deferred = nullptr,
        const tyloi::DenseTensor& input_exponents = {}) {
    using namespace tyloi;
    using namespace tyloi::cuda;
    const bool packed_input = input_exponents.defined();
    const auto* setting = env_lookup("TYLOINFER_GGUF_PREFILL_TENSORCORE");
    if ((setting && setting[0] != '1' && setting[0] != 'a') || !input.is_cuda() ||
        input.device() != weight.device() || input.scalar_type() != (packed_input ? kFloat16 : kFloat32) ||
        !input.is_contiguous() ||
        input.dim() != 2 || input.size(0) < 512 || input.size(0) > INT_MAX ||
        input.size(1) != weight.neuron_len || weight.neuron_len < 1 || weight.neuron_len > INT_MAX ||
        weight.out < 1 || weight.out > INT_MAX) return {};
    const std::int64_t tokens = input.size(0), columns = input.size(1), rows = weight.out;
    if (packed_input && (input_exponents.device() != input.device() || input_exponents.scalar_type() != kInt32 ||
        !input_exponents.is_contiguous() || input_exponents.numel() != tokens))
        return {};
    const auto device = input.get_device();
    const DeviceGuard guard(device);
    const auto stream = current_stream(device);
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(), &capture));
    if (capture != cudaStreamCaptureStatusNone || GraphAllocationPlan::warming() ||
        !gguf::grouped_f16_available(device) ||
        prefill_weight_rows(rows, std::uint64_t(columns) * 6 + std::uint64_t(tokens) * 2, 32, device) < rows) return {};

    const auto half = input.options().dtype(kFloat16), integers = input.options().dtype(kInt32);
    try {
        DenseTensor packed, row_exponents;
        if (packed_input) {
            packed = input;
            row_exponents = input_exponents;
        } else if (!gguf::GgufPrefillInputScope::lookup(input, packed, row_exponents)) {
            packed = empty({tokens, columns}, half);
            row_exponents = empty({tokens}, integers);
            gguf::pack_scaled_f16_unit(input.data_ptr<float>(), packed.data_ptr(),
                row_exponents.data_ptr<std::int32_t>(), tokens, columns, stream.stream());
        }
        auto image = empty({rows, columns}, half);
        auto column_exponents = empty({rows}, integers);
        {
            DenseTensor decoded;
            if constexpr (std::is_same_v<Weight, NintTensor>) decoded = dequant_nint_dense_f32(weight);
            else decoded = nvq_dequant(weight).to(kFloat32).contiguous();
            if (decoded.device() != input.device() || decoded.scalar_type() != kFloat32 ||
                !decoded.is_contiguous() || decoded.sizes() != IntArrayRef{rows, columns})
                throw std::runtime_error("scaled prefill decoder returned an incompatible weight matrix");
            gguf::pack_scaled_f16_unit(decoded.data_ptr<float>(), image.data_ptr(),
                column_exponents.data_ptr<std::int32_t>(), rows, columns, stream.stream());
        }
        auto product = empty({tokens, rows}, half);
        auto& blas = default_context(device)->blas(stream.stream());
        if (deferred) {
            gguf::prefill_gemm_f16_accum(packed.data_ptr(), image.data_ptr(), nullptr, product.data_ptr(),
                true, nullptr, nullptr, tokens, rows, columns, rows, blas.get(), stream.stream());
            *deferred = {std::move(product), std::move(row_exponents), std::move(column_exponents)};
            return {};
        }
        auto output = empty({tokens, rows}, input.options());
        gguf::prefill_gemm_f16_accum(packed.data_ptr(), image.data_ptr(), product.data_ptr(), output.data_ptr(),
            false, row_exponents.data_ptr<std::int32_t>(), column_exponents.data_ptr<std::int32_t>(),
            tokens, rows, columns, rows, blas.get(), stream.stream());
        return output;
    } catch (const MfqResourceExhausted&) {
        return {};
    }
}

// Execution policy stays here; the format operators only receive a profiler.
::tyloi::DenseTensor run_nint_linear(
        CudaProfiler& profiler,
        KlMmqState& kl_mmq,
        const tyloi::NintTensor& w,
        ::tyloi::DenseTensor x,
        MfqOptional<::tyloi::DenseTensor> gate,
        InputGateMode mode) {
    if (!gate.has_value() && kl_mmq.mode == KlMmqMode::Default)
        if (auto output = scaled_prefill(w, x); output.defined()) return output;
    if (!x.is_cuda() || !w.q8_zero || kl_mmq.mode == KlMmqMode::Default) {
        return gate.has_value()
            ? nint_matmul_input_mul(profiler, w, x, gate.value(), static_cast<int>(mode))
            : nint_matmul(profiler, w, x);
    }
    x = x.contiguous().to(::tyloi::kFloat16);
    if (gate.has_value()) {
        if (!(mode == InputGateMode::Sigmoid || mode == InputGateMode::Silu))
            throw std::runtime_error("NINT input gate mode must be sigmoid or SiLU");
        auto g = gate.value().contiguous().to(::tyloi::kFloat16);
        if (!(x.sizes() == g.sizes()))
            throw std::runtime_error("NINT x and gate shapes must match");
        x = mode == InputGateMode::Sigmoid ? x * ::tyloi::cuda::sigmoid(g) : x * ::tyloi::cuda::silu(g);
    }
    x = pad_last(x, w.neuron_len);
    const int64_t rows = x.size(0);
    if (!(rows > 0))
        throw std::runtime_error("KLD NINT8-0 MMQ requires activation rows");
    if (rows < 16) {
        x = ::tyloi::cuda::cat({x, ::tyloi::cuda::zeros(
            {16 - rows, x.size(1)}, x.options())}, 0).contiguous();
    }
    x = kl_mmq.prepare_activation(x);
    ++kl_mmq.dense_calls;
    auto result = profiler.measure("kld_mmq.nint8_zero.fp16", [&]() {
        return nint8_zero_mmq_f16_packed_cuda(
            w.q_packed, w.q8_zero_scale, x, w.neuron_len);
    });
    return rows < 16 ? result.narrow(0, 0, rows).contiguous() : result;
}

::tyloi::DenseTensor run_nvq_linear(
        CudaProfiler& profiler,
        KlMmqState& kl_mmq,
        const tyloi::NvqTensor& w,
        ::tyloi::DenseTensor x,
        MfqOptional<::tyloi::DenseTensor> gate,
        InputGateMode mode) {
    if (!gate.has_value() && kl_mmq.mode == KlMmqMode::Default)
        if (auto output = scaled_prefill(w, x); output.defined()) return output;
    if (!x.is_cuda() || kl_mmq.mode == KlMmqMode::Default) {
        return gate.has_value()
            ? nvq_matmul_input_mul(profiler, w, x, gate.value(), static_cast<int>(mode))
            : nvq_matmul(profiler, w, x);
    }
    x = x.contiguous().to(::tyloi::kFloat16);
    if (gate.has_value()) {
        if (!(mode == InputGateMode::Sigmoid || mode == InputGateMode::Silu))
            throw std::runtime_error("NVQ input gate mode must be sigmoid or SiLU");
        auto g = gate.value().contiguous().to(::tyloi::kFloat16);
        if (!(x.sizes() == g.sizes()))
            throw std::runtime_error("NVQ x and gate shapes must match");
        x = mode == InputGateMode::Sigmoid ? x * ::tyloi::cuda::sigmoid(g) : x * ::tyloi::cuda::silu(g);
    }
    x = pad_last(x, w.neuron_len);
    if (!(x.size(0) >= 16))
        throw std::runtime_error("KLD common NVQ MMQ requires at least 16 activation rows");
    x = kl_mmq.prepare_activation(x);
    ++kl_mmq.dense_calls;
    return profiler.measure("kld_mmq.nvq.fp16", [&]() {
        return nvq_gemm_f16_cuda(
            w.indices_packed, w.aux_packed, w.sub_scale_packed,
            w.neuron_scale, w.codebook, x, w.neuron_len, w.gs,
            w.sub_bits, w.kernel_format, w.sign_mode);
    });
}
