import jax
import jax.numpy as jnp

# 打印可用设备
print("Available devices:", jax.devices())

# 创建一个简单的计算，检查计算所使用的设备
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.dot(x, x)
print("Computation result:", y)
print("Computation device:", y.device_buffer.device())
