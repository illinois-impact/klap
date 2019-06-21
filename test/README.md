
# Instructions for Compiling and Running the Benchmarks

## Instructions for Compiling and Running All Benchmarks

To compile baseline versions: `make base`

To compile warp-granularity aggregation versions: `make aw`

To compile block-granularity aggregation versions: `make ab`

To compile grid-granularity aggregation versions: `make ag`

To compile all versions: `make all`

To run only the versions that have already been compiled: `make test`

To compile and run all versions: `make test-all`

After running the benchmarks, timing results will appear in `all.csv`

## Instructions for Compiling and Running an Individual Benchmark

Let ABC be the benchmark name.

Go into the benchmark directory: `cd ABC`

To compile baseline version: `make`

To compile warp-granularity aggregation version: `make abc.aw`

To compile block-granularity aggregation version: `make abc.ab`

To compile grid-granularity aggregation version: `make abc.ag`

To compile all versions: `make all`

To run only the versions that have already been compiled: `make test`

To compile and run all versions: `make test-all`

After running the benchmark, timing results will appear in `abc.csv`

