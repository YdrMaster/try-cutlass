# 试试 cutlass

## 配置

```xmake
xmake f --toolchain=cuda --cuda=$CUDA_ROOT -c -v
```

**根据实际硬件修改 [arch](/xmake.lua#L7)。**

```lua
add_cuflags("-arch=sm_xx")
```

## 编译

```xmake
xmake
```

## 运行

```xmake
xmake run gemm 8192 8192 8192
```
