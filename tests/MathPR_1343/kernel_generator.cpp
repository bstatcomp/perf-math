#define STAN_OPENCL
#define OPENCL_PLATFORM_ID 0
#define OPENCL_DEVICE_ID 0

#include <benchmark/benchmark.h>
#include <stan/math.hpp>

using namespace stan::math;
using namespace stan::math::opencl_kernels;
using namespace Eigen;

static void base1(benchmark::State& state) {
  int N = state.range(0);
  matrix_cl<double> a(N,N),b(N,N);

  for (auto _ : state) {
    matrix_cl<double> d = a + b;
  }
}
BENCHMARK(base1)->Args({100})->Args({1000})->Args({10000});

static void base2(benchmark::State& state) {
  int N = state.range(0);
  matrix_cl<double> a(N,N),b(N,N),c(N,N);

  for (auto _ : state) {
    matrix_cl<double> d = a + b - 3 * c;
  }
}
BENCHMARK(base2)->Args({100})->Args({1000})->Args({10000});


static void base3(benchmark::State& state) {
  int N = state.range(0);
  matrix_cl<double> a(N,N),b(N,N),c(N,N);

  for (auto _ : state) {
    matrix_cl<double> e = 2*a + 3*b - 4*c;
  }
}
BENCHMARK(base3)->Args({100})->Args({1000})->Args({10000});


static void base4(benchmark::State& state) {
  int N = state.range(0);
  matrix_cl<double> a(N,N),b(N,N),c(N,N);

  for (auto _ : state) {
    matrix_cl<double> d = a * b - 3 * c;
  }
}
BENCHMARK(base4)->Args({100})->Args({1000})->Args({10000});

#include <stan/math/opencl/kernel_generator/binary_operation.hpp>

static void kernel_generator1(benchmark::State& state) {
  int N = state.range(0);
  matrix_cl<double> a(N,N),b(N,N);
  matrix_cl<double> compile = a + b; //make sure the kernel is compiled before the benchmark

  for (auto _ : state) {
    matrix_cl<double> d = a + b;
  }
}
BENCHMARK(kernel_generator1)->Args({100})->Args({1000})->Args({10000});

static void kernel_generator2(benchmark::State& state) {
  int N = state.range(0);
  matrix_cl<double> a(N,N),b(N,N),c(N,N);
  matrix_cl<double> compile = a + b - 3 * c; //make sure the kernel is compiled before the benchmark

  for (auto _ : state) {
    matrix_cl<double> d = a + b - 3 * c;
  }
}
BENCHMARK(kernel_generator2)->Args({100})->Args({1000})->Args({10000});

static void kernel_generator3(benchmark::State& state) {
  int N = state.range(0);
  matrix_cl<double> a(N,N),b(N,N),c(N,N);
  matrix_cl<double> compile = 2*a + 3*b - 4*c; //make sure the kernel is compiled before the benchmark

  for (auto _ : state) {
    matrix_cl<double> d = 2*a + 3*b - 4*c;
  }
}
BENCHMARK(kernel_generator3)->Args({100})->Args({1000})->Args({10000});

static void kernel_generator4(benchmark::State& state) {
  int N = state.range(0);
  matrix_cl<double> a(N,N),b(N,N),c(N,N);
  matrix_cl<double> compile = a * b - 3 * c; //make sure the kernel is compiled before the benchmark

  for (auto _ : state) {
    matrix_cl<double> d = a * b - 3 * c;
  }
}
BENCHMARK(kernel_generator4)->Args({100})->Args({1000})->Args({10000});

BENCHMARK_MAIN();