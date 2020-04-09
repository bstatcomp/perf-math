#define STAN_OPENCL
#define OPENCL_DEVICE_ID 0
#define OPENCL_PLATFORM_ID 0
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <benchmark/benchmark.h>
#include <stan/math.hpp>

using namespace stan::math;

static void matvec_mul(benchmark::State& state) {
  matrix_cl<double> m(state.range(0), state.range(1));
  matrix_cl<double> v(state.range(1), 1);
  //  matrix_cl<double> res = m * v;
  matrix_cl<double> res(state.range(0),1);// = matrix_vector_multiply(m, v);
  //res.wait_for_write_events();

  for (auto _ : state) {
        res = m * v;
//    res = matrix_vector_multiply(m, v);
    res.wait_for_write_events();
  }
}
BENCHMARK(matvec_mul)
    ->Args({1000, 1000})
    ->Args({32, 30000})
    ->Args({30000, 32})
    ->Args({20000, 20000})
    ->Args({32, 600000})
    ->Args({600000, 32});

BENCHMARK_MAIN();
