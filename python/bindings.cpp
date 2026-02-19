#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <map>
#include <vector>
#include <cstring>

namespace py = pybind11;

// Forward declaration
class GpuArray;

// =============================================================================
// MetalContext — lightweight GPU context that loads metallib files
// =============================================================================
class MetalContext
{
public:
    MTL::Device *device;
    MTL::CommandQueue *queue;
    std::map<std::string, MTL::ComputePipelineState *> pipelines;

    MetalContext()
    {
        device = MTL::CreateSystemDefaultDevice();
        if (!device)
            throw std::runtime_error("No Metal device found");
        queue = device->newCommandQueue();
    }

    void loadLibrary(const std::string &path)
    {
        NS::Error *error = nullptr;
        auto nsPath = NS::String::string(path.c_str(), NS::ASCIIStringEncoding);
        MTL::Library *lib = device->newLibrary(nsPath, &error);
        if (!lib)
        {
            std::string msg = "Failed to load metallib: " + path;
            if (error)
                msg += " (" + std::string(error->description()->utf8String()) + ")";
            throw std::runtime_error(msg);
        }
        auto names = lib->functionNames();
        for (size_t i = 0; i < names->count(); i++)
        {
            auto nameNS = names->object(i)->description();
            auto fn = lib->newFunction(nameNS);
            auto pso = device->newComputePipelineState(fn, &error);
            fn->release();
            if (!pso)
                throw std::runtime_error("Failed to create pipeline for " +
                                         std::string(nameNS->utf8String()));
            pipelines[nameNS->utf8String()] = pso;
        }
        lib->release();
    }

    ~MetalContext()
    {
        for (auto &[k, v] : pipelines)
            v->release();
        queue->release();
        device->release();
    }
};

// =============================================================================
// GpuArray — zero-copy wrapper around a shared-memory Metal buffer
// =============================================================================
class GpuArray
{
public:
    MTL::Buffer *buffer;
    std::vector<size_t> shape;
    size_t size; // total number of floats

    // Allocate uninitialized buffer
    GpuArray(MetalContext &ctx, std::vector<size_t> shape_)
        : shape(shape_)
    {
        size = 1;
        for (auto s : shape)
            size *= s;
        buffer = ctx.device->newBuffer(size * sizeof(float), MTL::ResourceStorageModeShared);
    }

    // Allocate and fill with zeros
    static GpuArray zeros(MetalContext &ctx, std::vector<size_t> shape_)
    {
        GpuArray arr(ctx, shape_);
        std::memset(arr.buffer->contents(), 0, arr.size * sizeof(float));
        return arr;
    }

    // Create from numpy array (single copy in)
    static GpuArray fromNumpy(MetalContext &ctx, py::array_t<float> arr)
    {
        auto info = arr.request();
        std::vector<size_t> shape_(info.ndim);
        for (int i = 0; i < info.ndim; i++)
            shape_[i] = info.shape[i];
        GpuArray ga(ctx, shape_);
        std::memcpy(ga.buffer->contents(), info.ptr, ga.size * sizeof(float));
        return ga;
    }

    // Zero-copy numpy view into the shared memory
    py::array_t<float> numpy()
    {
        std::vector<py::ssize_t> py_shape(shape.begin(), shape.end());
        std::vector<py::ssize_t> strides(shape.size());
        py::ssize_t stride = sizeof(float);
        for (int i = shape.size() - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
        // Create a numpy array that points directly at the Metal buffer memory.
        // The GpuArray must stay alive while the numpy view is used.
        return py::array_t<float>(py_shape, strides, (float *)buffer->contents());
    }

    // Copy out to a new numpy array (safe — independent of buffer lifetime)
    py::array_t<float> toNumpy()
    {
        std::vector<py::ssize_t> py_shape(shape.begin(), shape.end());
        py::array_t<float> result(py_shape);
        std::memcpy(result.mutable_data(), buffer->contents(), size * sizeof(float));
        return result;
    }

    ~GpuArray()
    {
        if (buffer)
            buffer->release();
    }

    // Move-only semantics
    GpuArray(GpuArray &&other) noexcept
        : buffer(other.buffer), shape(std::move(other.shape)), size(other.size)
    {
        other.buffer = nullptr;
        other.size = 0;
    }
    GpuArray &operator=(GpuArray &&other) noexcept
    {
        if (this != &other)
        {
            if (buffer)
                buffer->release();
            buffer = other.buffer;
            shape = std::move(other.shape);
            size = other.size;
            other.buffer = nullptr;
            other.size = 0;
        }
        return *this;
    }
    GpuArray(const GpuArray &) = delete;
    GpuArray &operator=(const GpuArray &) = delete;
};

// =============================================================================
// Dispatch helpers
// =============================================================================

static void dispatch1D(MetalContext &ctx, const std::string &kernel,
                       std::vector<MTL::Buffer *> &buffers, size_t length)
{
    auto it = ctx.pipelines.find(kernel);
    if (it == ctx.pipelines.end())
        throw std::runtime_error("Unknown kernel: " + kernel);

    auto cmdBuf = ctx.queue->commandBuffer();
    auto enc = cmdBuf->computeCommandEncoder();
    enc->setComputePipelineState(it->second);
    for (size_t i = 0; i < buffers.size(); i++)
        enc->setBuffer(buffers[i], 0, i);

    NS::UInteger tgSize = it->second->maxTotalThreadsPerThreadgroup();
    if (tgSize > length)
        tgSize = length;
    enc->dispatchThreads(MTL::Size::Make(length, 1, 1),
                         MTL::Size::Make(tgSize, 1, 1));
    enc->endEncoding();
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();
}

static void dispatch2D(MetalContext &ctx, const std::string &kernel,
                       std::vector<MTL::Buffer *> &buffers, size_t width, size_t height)
{
    auto it = ctx.pipelines.find(kernel);
    if (it == ctx.pipelines.end())
        throw std::runtime_error("Unknown kernel: " + kernel);

    auto cmdBuf = ctx.queue->commandBuffer();
    auto enc = cmdBuf->computeCommandEncoder();
    enc->setComputePipelineState(it->second);
    for (size_t i = 0; i < buffers.size(); i++)
        enc->setBuffer(buffers[i], 0, i);

    // Use a square-ish threadgroup size
    NS::UInteger maxTg = it->second->maxTotalThreadsPerThreadgroup();
    NS::UInteger tgW = 1, tgH = 1;
    // Simple heuristic: try to get a reasonable 2D threadgroup
    for (NS::UInteger w = 1; w * w <= maxTg; w++)
        tgW = w;
    tgH = tgW;
    if (tgW > width)
        tgW = width;
    if (tgH > height)
        tgH = height;

    enc->dispatchThreads(MTL::Size::Make(width, height, 1),
                         MTL::Size::Make(tgW, tgH, 1));
    enc->endEncoding();
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();
}

// Helper: copy numpy array into a Metal buffer (for legacy API)
static MTL::Buffer *arrayToBuffer(MetalContext &ctx, py::array_t<float> arr)
{
    auto info = arr.request();
    size_t nbytes = info.size * sizeof(float);
    auto buf = ctx.device->newBuffer(nbytes, MTL::ResourceStorageModeShared);
    std::memcpy(buf->contents(), info.ptr, nbytes);
    return buf;
}

// Helper: copy Metal buffer back to numpy array (for legacy API)
static py::array_t<float> bufferToArray(MTL::Buffer *buf, std::vector<py::ssize_t> shape)
{
    size_t total = 1;
    for (auto s : shape)
        total *= s;
    py::array_t<float> result(shape);
    std::memcpy(result.mutable_data(), buf->contents(), total * sizeof(float));
    return result;
}

// =============================================================================
// Python module
// =============================================================================

PYBIND11_MODULE(_m1_gpu_ops, m)
{
    m.doc() = "Metal GPU operations for scientific computing on Apple Silicon";

    // --- MetalContext ---
    py::class_<MetalContext>(m, "MetalContext")
        .def(py::init<>())
        .def("load_library", &MetalContext::loadLibrary,
             "Load a .metallib file", py::arg("path"))
        .def_property_readonly("device_name", [](MetalContext &ctx)
                               { return std::string(ctx.device->name()->utf8String()); });

    // --- GpuArray ---
    py::class_<GpuArray>(m, "GpuArray")
        .def(py::init([](MetalContext &ctx, std::vector<size_t> shape)
                      { return GpuArray(ctx, shape); }),
             "Allocate uninitialized GPU buffer", py::arg("ctx"), py::arg("shape"))
        .def_static("zeros", &GpuArray::zeros,
                    "Allocate zero-filled GPU buffer", py::arg("ctx"), py::arg("shape"))
        .def_static("from_numpy", &GpuArray::fromNumpy,
                    "Create GPU buffer from numpy array (single copy)", py::arg("ctx"), py::arg("arr"))
        .def("numpy", &GpuArray::numpy,
             "Zero-copy numpy view into GPU shared memory (buffer must stay alive)")
        .def("to_numpy", &GpuArray::toNumpy,
             "Copy GPU buffer to a new numpy array")
        .def_readonly("shape", &GpuArray::shape)
        .def_readonly("size", &GpuArray::size);

    // =========================================================================
    // 1D operations (from 02-GeneralArrayOperations)
    // =========================================================================

    m.def(
        "add_arrays",
        [](MetalContext &ctx, py::array_t<float> x, py::array_t<float> y)
        {
            auto xInfo = x.request();
            auto yInfo = y.request();
            if (xInfo.size != yInfo.size)
                throw std::runtime_error("Array sizes must match");
            size_t n = xInfo.size;

            auto xBuf = arrayToBuffer(ctx, x);
            auto yBuf = arrayToBuffer(ctx, y);
            auto rBuf = ctx.device->newBuffer(n * sizeof(float), MTL::ResourceStorageModeShared);

            std::vector<MTL::Buffer *> bufs = {xBuf, yBuf, rBuf};
            dispatch1D(ctx, "add_arrays", bufs, n);

            auto result = bufferToArray(rBuf, {(py::ssize_t)n});
            xBuf->release();
            yBuf->release();
            rBuf->release();
            return result;
        },
        "GPU element-wise addition: result = x + y", py::arg("ctx"), py::arg("x"), py::arg("y"));

    m.def(
        "multiply_arrays",
        [](MetalContext &ctx, py::array_t<float> x, py::array_t<float> y)
        {
            auto xInfo = x.request();
            size_t n = xInfo.size;

            auto xBuf = arrayToBuffer(ctx, x);
            auto yBuf = arrayToBuffer(ctx, y);
            auto rBuf = ctx.device->newBuffer(n * sizeof(float), MTL::ResourceStorageModeShared);

            std::vector<MTL::Buffer *> bufs = {xBuf, yBuf, rBuf};
            dispatch1D(ctx, "multiply_arrays", bufs, n);

            auto result = bufferToArray(rBuf, {(py::ssize_t)n});
            xBuf->release();
            yBuf->release();
            rBuf->release();
            return result;
        },
        "GPU element-wise multiplication: result = x * y", py::arg("ctx"), py::arg("x"), py::arg("y"));

    m.def(
        "saxpy",
        [](MetalContext &ctx, float alpha, py::array_t<float> x, py::array_t<float> y)
        {
            size_t n = x.request().size;

            auto aBuf = ctx.device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
            *(float *)aBuf->contents() = alpha;
            auto xBuf = arrayToBuffer(ctx, x);
            auto yBuf = arrayToBuffer(ctx, y);
            auto rBuf = ctx.device->newBuffer(n * sizeof(float), MTL::ResourceStorageModeShared);

            std::vector<MTL::Buffer *> bufs = {aBuf, xBuf, yBuf, rBuf};
            dispatch1D(ctx, "saxpy", bufs, n);

            auto result = bufferToArray(rBuf, {(py::ssize_t)n});
            aBuf->release();
            xBuf->release();
            yBuf->release();
            rBuf->release();
            return result;
        },
        "GPU SAXPY: result = alpha * x + y", py::arg("ctx"), py::arg("alpha"), py::arg("x"), py::arg("y"));

    // =========================================================================
    // 2D operations (from 03-2DKernels)
    // =========================================================================

    m.def(
        "laplacian2d",
        [](MetalContext &ctx, py::array_t<float> X)
        {
            auto info = X.request();
            if (info.ndim != 2)
                throw std::runtime_error("Input must be 2D");
            size_t rows = info.shape[0];
            size_t cols = info.shape[1];
            size_t n = rows * cols;

            auto Xc = py::array_t<float>::ensure(X);
            auto xBuf = arrayToBuffer(ctx, Xc);
            auto rBuf = ctx.device->newBuffer(n * sizeof(float), MTL::ResourceStorageModeShared);
            std::memset(rBuf->contents(), 0, n * sizeof(float));

            std::vector<MTL::Buffer *> bufs = {xBuf, rBuf};
            dispatch2D(ctx, "laplacian2d", bufs, rows, cols);

            auto result = bufferToArray(rBuf, {(py::ssize_t)rows, (py::ssize_t)cols});
            xBuf->release();
            rBuf->release();
            return result;
        },
        "GPU 5-point Laplacian stencil on a 2D grid", py::arg("ctx"), py::arg("X"));

    m.def(
        "laplacian2d9p",
        [](MetalContext &ctx, py::array_t<float> X)
        {
            auto info = X.request();
            if (info.ndim != 2)
                throw std::runtime_error("Input must be 2D");
            size_t rows = info.shape[0];
            size_t cols = info.shape[1];
            size_t n = rows * cols;

            auto Xc = py::array_t<float>::ensure(X);
            auto xBuf = arrayToBuffer(ctx, Xc);
            auto rBuf = ctx.device->newBuffer(n * sizeof(float), MTL::ResourceStorageModeShared);
            std::memset(rBuf->contents(), 0, n * sizeof(float));

            std::vector<MTL::Buffer *> bufs = {xBuf, rBuf};
            dispatch2D(ctx, "laplacian2d9p", bufs, rows, cols);

            auto result = bufferToArray(rBuf, {(py::ssize_t)rows, (py::ssize_t)cols});
            xBuf->release();
            rBuf->release();
            return result;
        },
        "GPU 9-point Laplacian stencil on a 2D grid", py::arg("ctx"), py::arg("X"));

    // =========================================================================
    // Compute-heavy operations (from 04-Compute)
    // =========================================================================

    m.def(
        "mandelbrot",
        [](MetalContext &ctx, size_t width, size_t height,
           float x_min, float x_max, float y_min, float y_max, int max_iter)
        {
            size_t n = width * height;

            // Result buffer
            auto rBuf = ctx.device->newBuffer(n * sizeof(float), MTL::ResourceStorageModeShared);

            // Params: [x_min, x_max, y_min, y_max, max_iter_float]
            auto pBuf = ctx.device->newBuffer(5 * sizeof(float), MTL::ResourceStorageModeShared);
            float *params = (float *)pBuf->contents();
            params[0] = x_min;
            params[1] = x_max;
            params[2] = y_min;
            params[3] = y_max;
            params[4] = (float)max_iter;

            std::vector<MTL::Buffer *> bufs = {rBuf, pBuf};
            dispatch2D(ctx, "mandelbrot", bufs, width, height);

            auto result = bufferToArray(rBuf, {(py::ssize_t)height, (py::ssize_t)width});
            rBuf->release();
            pBuf->release();
            return result;
        },
        "GPU Mandelbrot set computation",
        py::arg("ctx"), py::arg("width"), py::arg("height"),
        py::arg("x_min") = -2.0f, py::arg("x_max") = 1.0f,
        py::arg("y_min") = -1.5f, py::arg("y_max") = 1.5f,
        py::arg("max_iter") = 1000);

    m.def(
        "nbody_step",
        [](MetalContext &ctx, py::array_t<float> pos_mass_np,
           py::array_t<float> velocities_np, float dt, float softening)
        {
            auto posInfo = pos_mass_np.request();
            auto velInfo = velocities_np.request();
            if (posInfo.ndim != 2 || posInfo.shape[1] != 4)
                throw std::runtime_error("pos_mass must be (N, 4)");
            if (velInfo.ndim != 2 || velInfo.shape[1] != 4)
                throw std::runtime_error("velocities must be (N, 4)");
            size_t N = posInfo.shape[0];

            // Copy data to GPU buffers
            auto posBuf = ctx.device->newBuffer(N * 4 * sizeof(float), MTL::ResourceStorageModeShared);
            auto velBuf = ctx.device->newBuffer(N * 4 * sizeof(float), MTL::ResourceStorageModeShared);
            auto accBuf = ctx.device->newBuffer(N * 4 * sizeof(float), MTL::ResourceStorageModeShared);
            std::memcpy(posBuf->contents(), posInfo.ptr, N * 4 * sizeof(float));
            std::memcpy(velBuf->contents(), velInfo.ptr, N * 4 * sizeof(float));

            // uint params: [N]
            auto paramBuf = ctx.device->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
            *(uint32_t *)paramBuf->contents() = (uint32_t)N;

            // float params for forces: [softening^2]
            auto fparamBuf = ctx.device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
            *(float *)fparamBuf->contents() = softening * softening;

            // Step 1: compute forces
            std::vector<MTL::Buffer *> forceBufs = {posBuf, accBuf, paramBuf, fparamBuf};
            dispatch1D(ctx, "nbody_forces", forceBufs, N);

            // float params for integrate: [dt]
            auto dtBuf = ctx.device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
            *(float *)dtBuf->contents() = dt;

            // Step 2: integrate (leapfrog kick-drift)
            std::vector<MTL::Buffer *> intBufs = {posBuf, velBuf, accBuf, dtBuf};
            dispatch1D(ctx, "nbody_integrate", intBufs, N);

            // Copy results back
            py::array_t<float> posOut({(py::ssize_t)N, (py::ssize_t)4});
            py::array_t<float> velOut({(py::ssize_t)N, (py::ssize_t)4});
            std::memcpy(posOut.mutable_data(), posBuf->contents(), N * 4 * sizeof(float));
            std::memcpy(velOut.mutable_data(), velBuf->contents(), N * 4 * sizeof(float));

            posBuf->release();
            velBuf->release();
            accBuf->release();
            paramBuf->release();
            fparamBuf->release();
            dtBuf->release();
            return py::make_tuple(posOut, velOut);
        },
        "GPU N-body: compute forces + leapfrog integrate (one step)",
        py::arg("ctx"), py::arg("pos_mass"), py::arg("velocities"),
        py::arg("dt") = 0.001f, py::arg("softening") = 0.01f);

    m.def(
        "nbody_simulate",
        [](MetalContext &ctx, py::array_t<float> pos_mass_np,
           py::array_t<float> velocities_np, float dt, float softening, int n_steps)
        {
            auto posInfo = pos_mass_np.request();
            auto velInfo = velocities_np.request();
            if (posInfo.ndim != 2 || posInfo.shape[1] != 4)
                throw std::runtime_error("pos_mass must be (N, 4)");
            if (velInfo.ndim != 2 || velInfo.shape[1] != 4)
                throw std::runtime_error("velocities must be (N, 4)");
            size_t N = posInfo.shape[0];

            // Allocate GPU buffers once
            auto posBuf = ctx.device->newBuffer(N * 4 * sizeof(float), MTL::ResourceStorageModeShared);
            auto velBuf = ctx.device->newBuffer(N * 4 * sizeof(float), MTL::ResourceStorageModeShared);
            auto accBuf = ctx.device->newBuffer(N * 4 * sizeof(float), MTL::ResourceStorageModeShared);
            std::memcpy(posBuf->contents(), posInfo.ptr, N * 4 * sizeof(float));
            std::memcpy(velBuf->contents(), velInfo.ptr, N * 4 * sizeof(float));

            auto paramBuf = ctx.device->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
            *(uint32_t *)paramBuf->contents() = (uint32_t)N;
            auto fparamBuf = ctx.device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
            *(float *)fparamBuf->contents() = softening * softening;
            auto dtBuf = ctx.device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
            *(float *)dtBuf->contents() = dt;

            auto it_forces = ctx.pipelines.find("nbody_forces");
            auto it_integ = ctx.pipelines.find("nbody_integrate");
            if (it_forces == ctx.pipelines.end() || it_integ == ctx.pipelines.end())
                throw std::runtime_error("nbody kernels not loaded");

            // Run n_steps on GPU without returning to Python
            for (int step = 0; step < n_steps; step++)
            {
                // Forces
                auto cmdBuf = ctx.queue->commandBuffer();
                auto enc = cmdBuf->computeCommandEncoder();
                enc->setComputePipelineState(it_forces->second);
                enc->setBuffer(posBuf, 0, 0);
                enc->setBuffer(accBuf, 0, 1);
                enc->setBuffer(paramBuf, 0, 2);
                enc->setBuffer(fparamBuf, 0, 3);
                NS::UInteger tg = it_forces->second->maxTotalThreadsPerThreadgroup();
                if (tg > N) tg = N;
                enc->dispatchThreads(MTL::Size::Make(N, 1, 1),
                                     MTL::Size::Make(tg, 1, 1));
                enc->endEncoding();

                // Integrate — encode in same command buffer
                enc = cmdBuf->computeCommandEncoder();
                enc->setComputePipelineState(it_integ->second);
                enc->setBuffer(posBuf, 0, 0);
                enc->setBuffer(velBuf, 0, 1);
                enc->setBuffer(accBuf, 0, 2);
                enc->setBuffer(dtBuf, 0, 3);
                tg = it_integ->second->maxTotalThreadsPerThreadgroup();
                if (tg > N) tg = N;
                enc->dispatchThreads(MTL::Size::Make(N, 1, 1),
                                     MTL::Size::Make(tg, 1, 1));
                enc->endEncoding();

                cmdBuf->commit();
                cmdBuf->waitUntilCompleted();
            }

            // Copy results back
            py::array_t<float> posOut({(py::ssize_t)N, (py::ssize_t)4});
            py::array_t<float> velOut({(py::ssize_t)N, (py::ssize_t)4});
            std::memcpy(posOut.mutable_data(), posBuf->contents(), N * 4 * sizeof(float));
            std::memcpy(velOut.mutable_data(), velBuf->contents(), N * 4 * sizeof(float));

            posBuf->release();
            velBuf->release();
            accBuf->release();
            paramBuf->release();
            fparamBuf->release();
            dtBuf->release();
            return py::make_tuple(posOut, velOut);
        },
        "GPU N-body simulation: n_steps of forces+integrate, data stays on GPU",
        py::arg("ctx"), py::arg("pos_mass"), py::arg("velocities"),
        py::arg("dt") = 0.001f, py::arg("softening") = 0.01f, py::arg("n_steps") = 100);

    m.def(
        "diffuse_steps",
        [](MetalContext &ctx, py::array_t<float> field_np, float dt, int n_steps)
        {
            auto info = field_np.request();
            if (info.ndim != 2)
                throw std::runtime_error("Input must be 2D");
            size_t rows = info.shape[0];
            size_t cols = info.shape[1];
            size_t n = rows * cols;

            // Two GPU buffers for ping-pong
            auto bufA = ctx.device->newBuffer(n * sizeof(float), MTL::ResourceStorageModeShared);
            auto bufB = ctx.device->newBuffer(n * sizeof(float), MTL::ResourceStorageModeShared);
            std::memcpy(bufA->contents(), info.ptr, n * sizeof(float));

            // Scalar dt buffer for SAXPY: result = dt * laplacian + field
            auto dtBuf = ctx.device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
            *(float *)dtBuf->contents() = dt;

            auto it_lap = ctx.pipelines.find("laplacian2d");
            auto it_saxpy = ctx.pipelines.find("saxpy");
            if (it_lap == ctx.pipelines.end() || it_saxpy == ctx.pipelines.end())
                throw std::runtime_error("laplacian2d and saxpy kernels must be loaded");

            // Temporary buffer for Laplacian result
            auto lapBuf = ctx.device->newBuffer(n * sizeof(float), MTL::ResourceStorageModeShared);

            MTL::Buffer *src = bufA;
            MTL::Buffer *dst = bufB;

            // Use a square-ish threadgroup for 2D dispatch
            NS::UInteger maxTg = it_lap->second->maxTotalThreadsPerThreadgroup();
            NS::UInteger tgW = 1;
            for (NS::UInteger w = 1; w * w <= maxTg; w++)
                tgW = w;
            NS::UInteger tgH = tgW;
            if (tgW > rows) tgW = rows;
            if (tgH > cols) tgH = cols;

            NS::UInteger tg1D = it_saxpy->second->maxTotalThreadsPerThreadgroup();
            if (tg1D > n) tg1D = n;

            for (int step = 0; step < n_steps; step++)
            {
                std::memset(lapBuf->contents(), 0, n * sizeof(float));

                auto cmdBuf = ctx.queue->commandBuffer();

                // Laplacian: lapBuf = laplacian(src)
                auto enc = cmdBuf->computeCommandEncoder();
                enc->setComputePipelineState(it_lap->second);
                enc->setBuffer(src, 0, 0);
                enc->setBuffer(lapBuf, 0, 1);
                enc->dispatchThreads(MTL::Size::Make(rows, cols, 1),
                                     MTL::Size::Make(tgW, tgH, 1));
                enc->endEncoding();

                // SAXPY: dst = dt * lapBuf + src
                enc = cmdBuf->computeCommandEncoder();
                enc->setComputePipelineState(it_saxpy->second);
                enc->setBuffer(dtBuf, 0, 0);
                enc->setBuffer(lapBuf, 0, 1);
                enc->setBuffer(src, 0, 2);
                enc->setBuffer(dst, 0, 3);
                enc->dispatchThreads(MTL::Size::Make(n, 1, 1),
                                     MTL::Size::Make(tg1D, 1, 1));
                enc->endEncoding();

                cmdBuf->commit();
                cmdBuf->waitUntilCompleted();

                std::swap(src, dst);
            }

            // src now points to the final result
            py::array_t<float> result({(py::ssize_t)rows, (py::ssize_t)cols});
            std::memcpy(result.mutable_data(), src->contents(), n * sizeof(float));

            bufA->release();
            bufB->release();
            lapBuf->release();
            dtBuf->release();
            return result;
        },
        "GPU multi-step diffusion: n_steps of (field += dt * laplacian(field)), data stays on GPU",
        py::arg("ctx"), py::arg("field"), py::arg("dt") = 0.01f, py::arg("n_steps") = 100);

    // =========================================================================
    // Elastic wave propagation (from 05-WavePropagation)
    // =========================================================================

    m.def(
        "elastic_wave_propagate",
        [](MetalContext &ctx,
           py::array_t<float> vp_np,
           py::array_t<float> vs_np,
           py::array_t<float> rho_np,
           int src_x, int src_z,
           py::array_t<float> wavelet_np,
           py::array_t<int> recv_x_np,
           py::array_t<int> recv_z_np,
           float dx, float dz, float dt,
           int snapshot_interval,
           int n_boundary)
        {
            // --- 1. Validate inputs ---
            auto vp_info = vp_np.request();
            auto vs_info = vs_np.request();
            auto rho_info = rho_np.request();
            if (vp_info.ndim != 2)
                throw std::runtime_error("vp must be 2D");
            size_t nx = vp_info.shape[0];
            size_t nz = vp_info.shape[1];
            size_t n = nx * nz;

            auto wav_info = wavelet_np.request();
            size_t nt = wav_info.shape[0];

            auto rx_info = recv_x_np.request();
            auto rz_info = recv_z_np.request();
            size_t nrec = rx_info.shape[0];
            int *recv_x = (int *)rx_info.ptr;
            int *recv_z = (int *)rz_info.ptr;

            float *vp_data = (float *)vp_info.ptr;
            float *vs_data = (float *)vs_info.ptr;
            float *rho_data = (float *)rho_info.ptr;
            float *wav_data = (float *)wav_info.ptr;

            // --- 2. Precompute material parameters on CPU ---
            size_t fbytes = n * sizeof(float);
            std::vector<float> lam2mu_h(n), lam_h(n), mu_xz_h(n, 0.0f);
            std::vector<float> b_x_h(n), b_z_h(n);

            for (size_t i = 0; i < nx; i++)
            {
                for (size_t j = 0; j < nz; j++)
                {
                    size_t idx = i * nz + j;
                    float r = rho_data[idx];
                    float vp2 = vp_data[idx] * vp_data[idx];
                    float vs2 = vs_data[idx] * vs_data[idx];
                    lam2mu_h[idx] = r * vp2;
                    lam_h[idx] = r * (vp2 - 2.0f * vs2);
                }
            }

            // mu_xz: harmonic mean of mu at 4 surrounding integer points
            for (size_t i = 0; i < nx - 1; i++)
            {
                for (size_t j = 0; j < nz - 1; j++)
                {
                    size_t idx = i * nz + j;
                    float mu00 = rho_data[idx] * vs_data[idx] * vs_data[idx];
                    float mu10 = rho_data[idx + nz] * vs_data[idx + nz] * vs_data[idx + nz];
                    float mu01 = rho_data[idx + 1] * vs_data[idx + 1] * vs_data[idx + 1];
                    float mu11 = rho_data[idx + nz + 1] * vs_data[idx + nz + 1] * vs_data[idx + nz + 1];
                    if (mu00 > 0 && mu10 > 0 && mu01 > 0 && mu11 > 0)
                        mu_xz_h[idx] = 4.0f / (1.0f / mu00 + 1.0f / mu10 + 1.0f / mu01 + 1.0f / mu11);
                }
            }

            // b_x: buoyancy at half-x points
            for (size_t i = 0; i < nx; i++)
            {
                for (size_t j = 0; j < nz; j++)
                {
                    size_t idx = i * nz + j;
                    if (i < nx - 1)
                        b_x_h[idx] = 0.5f * (1.0f / rho_data[idx] + 1.0f / rho_data[idx + nz]);
                    else
                        b_x_h[idx] = 1.0f / rho_data[idx];
                }
            }

            // b_z: buoyancy at half-z points
            for (size_t i = 0; i < nx; i++)
            {
                for (size_t j = 0; j < nz; j++)
                {
                    size_t idx = i * nz + j;
                    if (j < nz - 1)
                        b_z_h[idx] = 0.5f * (1.0f / rho_data[idx] + 1.0f / rho_data[idx + 1]);
                    else
                        b_z_h[idx] = 1.0f / rho_data[idx];
                }
            }

            // --- 3. Precompute damping (Cerjan sponge) ---
            std::vector<float> damp_h(n, 1.0f);
            if (n_boundary > 0)
            {
                float damping_factor = 0.015f;
                for (size_t i = 0; i < nx; i++)
                {
                    for (size_t j = 0; j < nz; j++)
                    {
                        int dist_x0 = (int)i;
                        int dist_x1 = (int)(nx - 1 - i);
                        int dist_z0 = (int)j;
                        int dist_z1 = (int)(nz - 1 - j);
                        int dist = dist_x0;
                        if (dist_x1 < dist) dist = dist_x1;
                        if (dist_z0 < dist) dist = dist_z0;
                        if (dist_z1 < dist) dist = dist_z1;
                        if (dist < n_boundary)
                        {
                            float t = damping_factor * (float)(n_boundary - dist);
                            damp_h[i * nz + j] = std::exp(-(t * t));
                        }
                    }
                }
            }

            // --- 4. Allocate GPU buffers ---
            auto vxBuf = ctx.device->newBuffer(fbytes, MTL::ResourceStorageModeShared);
            auto vzBuf = ctx.device->newBuffer(fbytes, MTL::ResourceStorageModeShared);
            auto sxxBuf = ctx.device->newBuffer(fbytes, MTL::ResourceStorageModeShared);
            auto szzBuf = ctx.device->newBuffer(fbytes, MTL::ResourceStorageModeShared);
            auto sxzBuf = ctx.device->newBuffer(fbytes, MTL::ResourceStorageModeShared);
            std::memset(vxBuf->contents(), 0, fbytes);
            std::memset(vzBuf->contents(), 0, fbytes);
            std::memset(sxxBuf->contents(), 0, fbytes);
            std::memset(szzBuf->contents(), 0, fbytes);
            std::memset(sxzBuf->contents(), 0, fbytes);

            auto lam2muBuf = ctx.device->newBuffer(fbytes, MTL::ResourceStorageModeShared);
            auto lamBuf = ctx.device->newBuffer(fbytes, MTL::ResourceStorageModeShared);
            auto muxzBuf = ctx.device->newBuffer(fbytes, MTL::ResourceStorageModeShared);
            auto bxBuf = ctx.device->newBuffer(fbytes, MTL::ResourceStorageModeShared);
            auto bzBuf = ctx.device->newBuffer(fbytes, MTL::ResourceStorageModeShared);
            std::memcpy(lam2muBuf->contents(), lam2mu_h.data(), fbytes);
            std::memcpy(lamBuf->contents(), lam_h.data(), fbytes);
            std::memcpy(muxzBuf->contents(), mu_xz_h.data(), fbytes);
            std::memcpy(bxBuf->contents(), b_x_h.data(), fbytes);
            std::memcpy(bzBuf->contents(), b_z_h.data(), fbytes);

            auto dampBuf = ctx.device->newBuffer(fbytes, MTL::ResourceStorageModeShared);
            std::memcpy(dampBuf->contents(), damp_h.data(), fbytes);

            auto paramBuf = ctx.device->newBuffer(5 * sizeof(float), MTL::ResourceStorageModeShared);
            float *params = (float *)paramBuf->contents();
            params[0] = dt;
            params[1] = dx;
            params[2] = dz;
            params[3] = (float)nx;
            params[4] = (float)nz;

            // --- 5. Lookup pipeline states ---
            auto it_stress = ctx.pipelines.find("stress_update");
            auto it_vel = ctx.pipelines.find("velocity_update");
            auto it_damp = ctx.pipelines.find("apply_damping");
            if (it_stress == ctx.pipelines.end() ||
                it_vel == ctx.pipelines.end() ||
                it_damp == ctx.pipelines.end())
                throw std::runtime_error(
                    "Wave propagation kernels not loaded. "
                    "Load the 05-WavePropagation metallib first.");

            // --- 6. Prepare output storage ---
            std::vector<float> seis_vx(nrec * nt, 0.0f);
            std::vector<float> seis_vz(nrec * nt, 0.0f);

            size_t n_snaps = (snapshot_interval > 0) ? (nt / snapshot_interval) : 0;
            std::vector<float> snap_vx(n_snaps * n, 0.0f);
            std::vector<float> snap_vz(n_snaps * n, 0.0f);

            // --- 7. Compute threadgroup sizes ---
            NS::UInteger maxTg2D = it_stress->second->maxTotalThreadsPerThreadgroup();
            NS::UInteger tgW = 1;
            for (NS::UInteger w = 1; w * w <= maxTg2D; w++)
                tgW = w;
            NS::UInteger tgH = tgW;
            if (tgW > nx) tgW = nx;
            if (tgH > nz) tgH = nz;

            NS::UInteger tg1D = it_damp->second->maxTotalThreadsPerThreadgroup();
            if (tg1D > n) tg1D = n;

            // --- 8. Time-stepping loop ---
            size_t snap_count = 0;
            float *vx_ptr = (float *)vxBuf->contents();
            float *vz_ptr = (float *)vzBuf->contents();

            for (size_t t = 0; t < nt; t++)
            {
                // 8a. Source injection (CPU write to shared memory)
                size_t src_idx = (size_t)src_x * nz + (size_t)src_z;
                ((float *)sxxBuf->contents())[src_idx] += wav_data[t];
                ((float *)szzBuf->contents())[src_idx] += wav_data[t];

                // 8b. Stress update (GPU)
                auto cmdBuf = ctx.queue->commandBuffer();
                auto enc = cmdBuf->computeCommandEncoder();
                enc->setComputePipelineState(it_stress->second);
                enc->setBuffer(vxBuf, 0, 0);
                enc->setBuffer(vzBuf, 0, 1);
                enc->setBuffer(sxxBuf, 0, 2);
                enc->setBuffer(szzBuf, 0, 3);
                enc->setBuffer(sxzBuf, 0, 4);
                enc->setBuffer(lam2muBuf, 0, 5);
                enc->setBuffer(lamBuf, 0, 6);
                enc->setBuffer(muxzBuf, 0, 7);
                enc->setBuffer(paramBuf, 0, 8);
                enc->dispatchThreads(MTL::Size::Make(nx, nz, 1),
                                     MTL::Size::Make(tgW, tgH, 1));
                enc->endEncoding();

                // 8c. Velocity update (GPU, same command buffer)
                enc = cmdBuf->computeCommandEncoder();
                enc->setComputePipelineState(it_vel->second);
                enc->setBuffer(sxxBuf, 0, 0);
                enc->setBuffer(szzBuf, 0, 1);
                enc->setBuffer(sxzBuf, 0, 2);
                enc->setBuffer(vxBuf, 0, 3);
                enc->setBuffer(vzBuf, 0, 4);
                enc->setBuffer(bxBuf, 0, 5);
                enc->setBuffer(bzBuf, 0, 6);
                enc->setBuffer(paramBuf, 0, 7);
                enc->dispatchThreads(MTL::Size::Make(nx, nz, 1),
                                     MTL::Size::Make(tgW, tgH, 1));
                enc->endEncoding();

                // 8d. Apply damping to all 5 fields (GPU, same command buffer)
                MTL::Buffer *fields[5] = {vxBuf, vzBuf, sxxBuf, szzBuf, sxzBuf};
                for (int f = 0; f < 5; f++)
                {
                    enc = cmdBuf->computeCommandEncoder();
                    enc->setComputePipelineState(it_damp->second);
                    enc->setBuffer(fields[f], 0, 0);
                    enc->setBuffer(dampBuf, 0, 1);
                    enc->dispatchThreads(MTL::Size::Make(n, 1, 1),
                                         MTL::Size::Make(tg1D, 1, 1));
                    enc->endEncoding();
                }

                cmdBuf->commit();
                cmdBuf->waitUntilCompleted();

                // 8e. Record receivers (CPU read from shared memory)
                for (size_t r = 0; r < nrec; r++)
                {
                    size_t ridx = (size_t)recv_x[r] * nz + (size_t)recv_z[r];
                    seis_vx[r * nt + t] = vx_ptr[ridx];
                    seis_vz[r * nt + t] = vz_ptr[ridx];
                }

                // 8f. Save snapshots
                if (snapshot_interval > 0 &&
                    ((t + 1) % snapshot_interval == 0) &&
                    snap_count < n_snaps)
                {
                    std::memcpy(&snap_vx[snap_count * n], vx_ptr, fbytes);
                    std::memcpy(&snap_vz[snap_count * n], vz_ptr, fbytes);
                    snap_count++;
                }
            }

            // --- 9. Copy results to numpy ---
            py::array_t<float> seis_vx_out({(py::ssize_t)nrec, (py::ssize_t)nt});
            py::array_t<float> seis_vz_out({(py::ssize_t)nrec, (py::ssize_t)nt});
            std::memcpy(seis_vx_out.mutable_data(), seis_vx.data(),
                        nrec * nt * sizeof(float));
            std::memcpy(seis_vz_out.mutable_data(), seis_vz.data(),
                        nrec * nt * sizeof(float));

            py::array_t<float> snap_vx_out(
                {(py::ssize_t)snap_count, (py::ssize_t)nx, (py::ssize_t)nz});
            py::array_t<float> snap_vz_out(
                {(py::ssize_t)snap_count, (py::ssize_t)nx, (py::ssize_t)nz});
            if (snap_count > 0)
            {
                std::memcpy(snap_vx_out.mutable_data(), snap_vx.data(),
                            snap_count * n * sizeof(float));
                std::memcpy(snap_vz_out.mutable_data(), snap_vz.data(),
                            snap_count * n * sizeof(float));
            }

            // --- 10. Release GPU buffers ---
            vxBuf->release();
            vzBuf->release();
            sxxBuf->release();
            szzBuf->release();
            sxzBuf->release();
            lam2muBuf->release();
            lamBuf->release();
            muxzBuf->release();
            bxBuf->release();
            bzBuf->release();
            dampBuf->release();
            paramBuf->release();

            return py::make_tuple(seis_vx_out, seis_vz_out, snap_vx_out, snap_vz_out);
        },
        "GPU 2D elastic wave propagation (Virieux staggered-grid scheme)",
        py::arg("ctx"),
        py::arg("vp"), py::arg("vs"), py::arg("rho"),
        py::arg("src_x"), py::arg("src_z"),
        py::arg("wavelet"),
        py::arg("recv_x"), py::arg("recv_z"),
        py::arg("dx"), py::arg("dz"), py::arg("dt"),
        py::arg("snapshot_interval") = 0,
        py::arg("n_boundary") = 20);
}
