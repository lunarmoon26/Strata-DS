# CPU expert interface only

This port retains only `moe_mul1.h` and `torch_shim.h`, the compatibility types
used by the shared DeepSeek core. The optimized CPU implementation is **not
included or linked**. Unexpected CPU expert operations throw through
`src/ds41/resident_cpu_guard.cpp`; experts are never silently dropped.

The interface was adapted by Jackwwg83/Strata-DS from exllamav3 v1.5.4 at
`16a49792a3c93d8432d72e6c4bce800841566577`. It is carried from Strata-DS at
`a46eeba2905de9759a580f830f9377b8ef27ca9b`. MIT license, copyright Turboderp;
see `LICENSE`. The resident engine requires a two-device split before opening
the pack or calling CUDA.
