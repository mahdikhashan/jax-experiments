import os
# artificially set the number of cpu devices to 8
# my device mac M1 
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

import jax

print(jax.devices("cpu"))
