# M1 GPUs for (scientific) computations in C++
## Outperforming OpenMP using Metal
### Lars Gebraad, 2022

This story of working with M1 chips is an amalgation of various Apple documentations. As
a scientific programmer it is a bit complicated to work on the new Apple M1 Macbooks;
CUDA doesn't work on these blazing fast chips! It would be cool if we can offload heavy
physics simulations to the GPU, they've shown that they are quite capable. We'll start
off slow, working out the basics of array operations on the GPU, and hopefully end up at
some proper fast physics!


## Introduction to Apple's Metal

Luckily for people who jumped on the new M1 chips: you can program rather easily for
this chip using Apple's [**Metal**](https://developer.apple.com/metal/) and it's
programming language MSL, Metal Shading Language.

Our second "luckily", MSL is C++ based. This is cool, because my scientific code is in
C++, I'm not going near Fortran.  MSL promises that we can compile our computational
kernels (or shaders as they're called in MSL) to fast code, and use them on heterogeneous
systems that support Metal. Here is a sample kernel written in MSL that adds two arrays
together:
```C++
kernel void add_arrays(device const float* A,
                       device const float* B,
                       device float* C,
                       uint index [[thread_position_in_grid]])
{
    C[index] = A[index] + B[index];
}
```
Looks a lot like CUDA right? Great! If you don't understand it yet, tag along, and you
will at the end. Let's try to get this to work on a MacBook and compare it against
OpenMP and serial implementations.

Metal has a bunch of other graphics-oriented functionality, but for scientific
programming we leave those be for now.

## 0. Readying your system to follow along
I make it a point not to use Xcode. Nothing intrinsically against this piece of
software, but to start out with, it is a lot more useful to me to see how dependencies
work without using all Xcode's handholding. I'll compile all binaries using LLVM's
homebrewed CLang++:
```bash
brew install llvm
```
This has the added benefit of allowing me to use OpenMP on the M1 chip. Make sure the library files for OMP are installed:
```bash
brew install libomp
```
However, we will need the SDK's that come with Xcode to be able to compile for MacBooks.
I advise installing Xcode through your Mac's app store.


## 1. Performing Calculations on a GPU using Metal

Our first stop on the world wide web is [Apple's own calculations on GPU](https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu?language=objc), titled **Performing Calculations on a GPU using Metal**.
Exactly what we need! Until... you see that the interface to work with Metal is
available in Swift and Objective-C. Not the typical dialects used in the lands of
Geophysics or other exact sciences.

The website does give a very good overview on getting kernels to work though. I
recommend sitting down and reading the text, even if Objective-C is not your cup of tea.
I never worked in it, but I still found the article helpful to understand MSL concepts.
Among other things, we get the fun anecdote:
> In Metal, code that runs on GPUs is called a shader, because historically they were first used to calculate colors in 3D graphics.

Additionally, we learn a few other things:
- Metal dynamically loads your shader library at runtime. That means that your
applications and shaders are separately compiled.
- Metal orchestrates computations usings command queues and command buffers. This allows
for asynchronous and heterogeneous operations, much like CUDA.
- The design of parallel loops works much like CUDA, where the kernel is a single function called with an index.
- Metal has different types of data buffers, that are differently exposed to the GPU and
CPU. For ease of this tutorial `MTLResourceStorageModeShared` seems very appropriate. We
can use data in memory for both GPU and CPU computations!

## 2. Using Metal from C++, using metal-cpp

A bit more targeted web-surfing reveals [another Apple page, geared towards running graphical Metal applications from C++](https://developer.apple.com/metal/cpp/), titled **Getting started with Metal-cpp**.
Those highlights do their name justice, i.e.:
- Alternative to Objective-C;
- No measureable overhead.

This manual is not geared towards the scientific computation that we were interested in,
but it does allow us to get started with Metal in C++. One could follow the instructions
to download `metal-cpp_macOS12_iOS15.zip`, however, I was (legally, courtesy of the Apache license) able to include the relevant code in [this repository](https://github.com/larsgeb/m1-gpu-cpp). The interesting bits for us are:
- `metal-cpp/Metal`
- `metal-cpp/Foundation`
- `metal-cpp/QuartzCore`

These folders contain the relevant headers exposing Metal's interface to
C++.

We could probaly use these to translate the Objective-C code by hand ...

## 3. Translating the Objective-C without understanding! 

I'm not going to learn Objective-C just to translate a bit of code. The 
programmer in me says that there is a more efficient way to rewrite
`Performing Calculations on a GPU using Metal` into useable C++. Autocompletion in
VSCode would be a perfect shortcut, no? Additionally, both Objective-C and C++ contain a
C in their name. They must be extremely similar. ... Right?

Specifically, we need to get the following files to work in a C++ implementation:

- `main.m`, Looks like a main function from C++.
- `MetalAdder.h`, Header files for a class, apparently.
- `MetalAdder.m`, Body of the class, I hope.
- `add.metal`, the MSL code! We've seen this before, and it's definitely the least intimidating.

### Translating `main.m`

The Objective-C code main function (`main.m`) starts confident, coming in hot with stuff I've never seen before:

```objc
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "MetalAdder.h"

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();

        // Create the custom object used to encapsulate the Metal code.
        // Initializes objects to communicate with the GPU.
        MetalAdder* adder = [[MetalAdder alloc] initWithDevice:device];
        
        // Create buffers to hold data
        [adder prepareData];
        
        // Send a command to the GPU to perform the calculation.
        [adder sendComputeCommand];

        NSLog(@"Execution finished");
    }
    return 0;
}
```
Seems like it creates a pointer to a computation device (such as a GPU, defined in `Metal.h`), and then passes that to a constructor for `MetalAdder` (that we haven't defined yet). Next it runs a few functions associated with this object.

Using some smart autocompletion to figure out how to create the Metal device, and
filling in the arbitrary gaps (arbitrary as we still have to define the MetalAdder
class), we end up with some C++ code that looks like this:

```C++
#include <iostream>
#include <omp.h>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"

#include "MetalAdder.hpp"

int main(int argc, char *argv[])
{
    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    MetalAdder *adder = new MetalAdder(device);

    adder->sendComputeCommand(); 
    adder->verifyResults();
}
```

Additionally, I've added IOStream and OMP to facilitate output and multicore stuff we'll
do to profile all the code.

### Translating `MetalAdder.m` and `MetalAdder.h`

The `MetalAdder` class is a way to keep track of data, command, etc. that are relevant
to using the GPU. I'll try to one-on-one translate this to C++ from Apple's tutorial,
but in one or two places I optimized the code to C++ standards. The final result is 
`MetalAdder.cpp` and `MetalAdder.hpp`.

Let us first have a look at the **constructor**. In Objective-C, it's signature is the following:

```objc
- (instancetype) initWithDevice: (id<MTLDevice>) device
{
    // ... body ...
}
```

meaning that it takes a pointer to an `MTLDevice`. Using VSCode, we realize that in the
C++ headers, all types that lead with MTL, such as `MTLDevice` are translated to
with a leading MTL namespace: `MTL::Device`. Additionally, the Ojective-C constructor
tests for Metal errors by making sure none of the created objects turn out to be `nil`.
The equivalent of this in C++ is to check against `nullptr`s.

Functionally, what happens next in the constructor is the following:

- **Loading the Metal library containing our shaders.** The C++ equivalent is radily
found using your editors code-completion.

Objective-C
```objc
id<MTLLibrary> defaultLibrary = [_mDevice newDefaultLibrary];
```
C++
```cpp
MTL::Library *defaultLibrary = _mDevice->newDefaultLibrary();
```

- **Loading a specific shader based on its name.** This is the biggest translation
mismatch, as the C++ method newFunctionWithName doesn't exist, and it's equivalent
doesn't accept `const char *`, only it's own implementation of strings.

Objective-C
```objc
id<MTLFunction> addFunction = [defaultLibrary newFunctionWithName:@"add_arrays"];
```
C++
```cpp
auto str = NS::String::string("add_arrays", NS::ASCIIStringEncoding);
MTL::Function *addFunction = defaultLibrary->newFunction(str);
```
- **Creating a pipeline state object.** This translates just like loading the default
library.
- **Creating a command queue object.** This also translates without issue.
- **Prepare the test data.** This is simply done by a (yet-to-write) method of the class
we are writing.

To understand exactly what the created objects do, refer to the Apple tutorial written
for Objective-C.

Most **other class methods** are translated one-to-one much the same way. One of the
major differences (inconveniences, more like) between the original Objective-C and the
C++ implementation can be initially found in `generateRandomFloatData`. This function
populates arbirtrary buffers with random data. To do this, it needs to set the values of
the buffer one by one, accessing these from the CPU.  In the Objective-C implementation,
whenever we want to access the buffer's data, we obtain a pointer to the start of the
buffer, and loop over it by pointer arithmetic:

```objc
- (void) generateRandomFloatData: (id<MTLBuffer>) buffer
{
    float* dataPtr = buffer.contents;

    for (unsigned long index = 0; index < arrayLength; index++)
    {
        dataPtr[index] = (float)rand()/(float)(RAND_MAX);
    }
}
```

In C++, one is not able to implicitly cast this pointer. The return type of
`buffer->contents()` is a `void *`, i.e. a pointer to any type of object. For safety,
one needs to explicitly cast this pointer to a `float *`.

```cpp
void MetalAdder::generateRandomFloatData(MTL::Buffer *buffer)
{
    // The pointer needs to be explicitly cast in C++, a difference from
    // Objective-C.
    float *dataPtr = (float *)buffer->contents();

    for (unsigned long index = 0; index < arrayLength; index++)
    {
        dataPtr[index] = (float)rand() / (float)(RAND_MAX);
    }
}
```

Pfew, getting quite close there to actually programming! This slight difference in
Objective-C w.r.t. C++ pops up again in `verifyResults`, and basically whenever we try
to access the buffer manually.

To keep it tidy, I moved all declarations to the `MetalAdder.h` (but mostly because I didn't figure out how to have some declaration *not* in the header).

## 4. Compiling metal-cpp programs.

We avoid looking at documentation some more, and by messing around with the make
command in the **Getting started with Metal-cpp** project reveals the CLang++
`includes` relevant to get Metal to work in C++:
```bash
larsgebraad@macbook:~$ pwd
/Users/larsgebraad/Downloads/LearnMetalCPP

larsgebraad@macbook:~$ make
clang++ -Wall -std=c++17 -I./metal-cpp -I./metal-cpp-extensions -fno-objc-arc -O2  -framework Metal -framework Foundation -framework Cocoa -framework CoreGraphics -framework MetalKit  learn-metal/00-window/00-window.o -o build/00-window
```

It seems we need to include the `metal-cpp`, `metal-cpp-extensions` folders, as well as
hooking the frameworks into CLang. `CoreGraphics` doesn't seem like something we'd need
in an terminal based application, as does Cocoa. After some tweaking, the bare
necessities to compile a command line program with Metal seems to be:
```console
larsgebraad@macbook:~$ clang++ -std=c++17 -I./metal-cpp -O2 \
    -framework Metal -framework Foundation -framework MetalKit \
    whatever.cpp
```
This seems like a good start for compiling our Metal program.

I skip my system-wide `clang++` in favour of `/opt/homebrew/opt/llvm/bin/clang++`, which
allows me to easily include OpenMP libraries, e.g.:
```console
larsgebraad@macbook:~$ /opt/homebrew/opt/llvm/bin/clang++ \
    -L/opt/homebrew/opt/libomp/lib -fopenmp some-openmp.cpp
```

Now, to actually compile our Metal+OpenMP application, we run:
```console
larsgebraad@macbook:~$ /opt/homebrew/opt/llvm/bin/clang++ \
    -std=c++17 -stdlib=libc++ -O2 \
    -L/opt/homebrew/opt/libomp/lib -fopenmp \
    -I./metal-cpp \
    -fno-objc-arc \
    -framework Metal -framework Foundation -framework MetalKit \
    -g 01-MetalAdder/main.cpp 01-MetalAdder/MetalAdder.cpp  -o 01-MetalAdder/benchmark.x
```

If one were to try out this executable, we'd find the following:

```console
larsgebraad@macbook:~$ ./01-MetalAdder/benchmark.x

Failed to find the default library.
[1]    13767 segmentation fault  ./benchmark.x
```

It seems that our GPU code itself is not compiled yet, as this is not standard when compiling the CPU code. To do this, we follow the instructions [of yet another Apple documentation website](https://developer.apple.com/documentation/metal/shader_libraries/building_a_library_with_metal_s_command-line_tools?language=objc), titled **Building a Library with Metal's Command-Line Tools** and geared towards Objective-C. 

This is where our installation of Xcode is relevant; we need to use the command line
tools and SDKs from Xcode to compile our gpu code:

```console
larsgebraad@macbook:~$ xcrun -sdk macosx metal -c add.metal -o MyLibrary.air  

larsgebraad@macbook:~$ xcrun -sdk macosx metallib MyLibrary.air -o default.metallib
```

The final name of the `.metallib` file is important, as our executable is only searching for the `default` library. This behaviour can be adapted in the constructor of `MetalAdder`.

## 5. Benchmarking against serial and OpenMP code

Now that our GPU code is compiled, we are ready to run a full benchmark. In `main.cpp`, additional serial and OpenMP implementations of this array addition are defined. By running the resulting `benchmark.x`, we get the following impressive results:


> System specs:
> 
> - 2021 MacBook pro
> - M1 Max, 10‑Core CPU, 32‑Core GPU und 16‑Core Neural Engine
> - 32 GB RAM

```console
larsgebraad@macbook:~$ ./01-MetalAdder/benchmark.x

Metal (GPU) code performance: 
Average time: 803.566ms +/- 48.427ms

Serial code performance: 
Average time: 2439.92ms +/- 74.4422ms

OpenMP (1 threads) code performance: 
Average time: 2427.24ms +/- 15.666ms

OpenMP (2 threads) code performance: 
Average time: 1315.2ms +/- 76.5438ms

OpenMP (3 threads) code performance: 
Average time: 1684.19ms +/- 46.0139ms

OpenMP (4 threads) code performance: 
Average time: 1339.81ms +/- 99.9749ms

OpenMP (5 threads) code performance: 

...

OpenMP (10 threads) code performance: 
Average time: 1756.53ms +/- 640.482ms
```

Weirdly enough, OpenMP is faster on even thread counts. Unsurprisingly, the serial code is the slowest. On this simple array addition problem, using Metal allows for a 3x speed-up with respect to the CPU single thread, and a 1.6x speedup to the fastest OpenMP configuration!

I suspect these numbers will be more dramatic when the computational kernels are more involved, but this we'll see later.