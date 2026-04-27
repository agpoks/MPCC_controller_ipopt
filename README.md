# MPCC Controller (Ipopt)

This project runs an MPCC controller in C++ using CasADi + Ipopt.

## Build and run

From the project root:

```bash
./run.sh
```

Or manually:

```bash
cmake -S . -B build_hsl -DCMAKE_BUILD_TYPE=Release
cmake --build build_hsl -- -j"$(nproc)"
./build_hsl/MPCC_controller_cpp
```

## Enable HSL linear solvers (MA57/MA86/MA97)

This project supports Ipopt linear solver selection through environment variables:

- `MPC_IPOPT_LINEAR_SOLVER` (examples: `ma57`, `ma86`, `ma97`, `mumps`)
- `MPC_IPOPT_HSL_LIB` (optional explicit path to HSL shared library)

### 1) Build ThirdParty-HSL (one-time)

If you already built it for another project, you can reuse it.

```bash
cd ~/ThirdParty-HSL
./configure
make -j"$(nproc)"
```

Expected library path:

```bash
~/ThirdParty-HSL/.libs/libcoinhsl.so
```

### 2) Build this project

```bash
cd ~/github/MPCC_controller_ipopt
cmake -S . -B build_hsl -DCMAKE_BUILD_TYPE=Release
cmake --build build_hsl -- -j"$(nproc)"
```

### 3) Run with MA57

```bash
cd ~/github/MPCC_controller_ipopt
export LD_LIBRARY_PATH="$HOME/ThirdParty-HSL/.libs:${LD_LIBRARY_PATH:-}"
export MPC_IPOPT_LINEAR_SOLVER=ma57
./build_hsl/MPCC_controller_cpp
```

### 4) Optional explicit HSL library path

```bash
export MPC_IPOPT_HSL_LIB="$HOME/ThirdParty-HSL/.libs/libcoinhsl.so"
```

## Fallback to MUMPS

If HSL is not available, use MUMPS:

```bash
export MPC_IPOPT_LINEAR_SOLVER=mumps
./build_hsl/MPCC_controller_cpp
```

## Troubleshooting

### Error: `return_status is 'Invalid_Option'`

Most common causes:

1. `MPC_IPOPT_LINEAR_SOLVER` is set to `ma57`/`ma86`/`ma97`, but HSL library is not visible at runtime.
2. `LD_LIBRARY_PATH` does not include `~/ThirdParty-HSL/.libs`.
3. Wrong `MPC_IPOPT_HSL_LIB` path.

Quick check:

```bash
ls -l ~/ThirdParty-HSL/.libs/libcoinhsl.so
echo "$LD_LIBRARY_PATH"
```

### Validate with one command

```bash
cd ~/github/MPCC_controller_ipopt && \
export LD_LIBRARY_PATH="$HOME/ThirdParty-HSL/.libs:${LD_LIBRARY_PATH:-}" && \
export MPC_IPOPT_LINEAR_SOLVER=ma57 && \
./build_hsl/MPCC_controller_cpp
```
