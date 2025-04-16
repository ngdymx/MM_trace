# MM_trace

# Using CHESSCC or PEANO?
To select the environment, modify the `ENABLE_CHESSCC` flag in the **Makefile**:
- Set `ENABLE_CHESSCC` to `1` to enable **CHESSCC**.
- Set it to `0` to use **PEANO**.

# Kernel

```
make kernel
```

It will generate mm.o.

# Bitstream

```
make bitstream
```

It will generate xclbin file.

# Instruction
```
make instructions
```

It will generate txt file.

# Host

```
make host
```

It will generate run.exe.

# Trace
```
make trace
```

It will generate json file.
