from collections.abc import Sequence
import jax.numpy as jnp
from jax._src.lax import control_flow as lax_control_flow
from jax._src.numpy.util import check_arraylike, promote_dtypes_inexact
from jax._src.typing import Array, ArrayLike, DTypeLike
from jax.random import binomial, split

RealArray = ArrayLike
IntegerArray = ArrayLike
DTypeLikeInt = DTypeLike
DTypeLikeUInt = DTypeLike
DTypeLikeFloat = DTypeLike
Shape = Sequence[int]

def multinomial(
    key: Array,
    n: RealArray,
    p: RealArray,
    *,
    shape: Shape | None = None,
    dtype: DTypeLikeFloat = float,
    unroll: int | bool = 1,
):
  r"""Sample from a multinomial distribution.

  The probability mass function is

  .. math::
      f(x;n,p) = \frac{n!}{x_1! \ldots x_k!} p_1^{x_1} \ldots p_k^{x_k}

  Args:
    key: PRNG key.
    n: number of trials. Should have shape broadcastable to ``p.shape[:-1]``.
    p: probability of each outcome, with outcomes along the last axis.
    shape: optional, a tuple of nonnegative integers specifying the result batch
      shape, that is, the prefix of the result shape excluding the last axis.
      Must be broadcast-compatible with ``p.shape[:-1]``. The default (None)
      produces a result shape equal to ``p.shape``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    unroll: optional, unroll parameter passed to :func:`jax.lax.scan` inside the
      implementation of this function.

  Returns:
    An array of counts for each outcome with the specified dtype and with shape
      ``p.shape`` if ``shape`` is None, otherwise ``shape + (p.shape[-1],)``.
  """

  check_arraylike("multinomial", n, p)
  n, p = promote_dtypes_inexact(n, p)

  if shape is None:
    shape = p.shape
  n = jnp.broadcast_to(n, shape[:-1])
  p = jnp.broadcast_to(p, shape)

  def f(remainder, ratio_key):
    ratio, key = ratio_key
    count = binomial(key, remainder, ratio.clip(0, 1), dtype=remainder.dtype)
    return remainder - count, count

  p = jnp.moveaxis(p, -1, 0)

  remaining_probs = lax_control_flow.cumsum(p, 0, reverse=True)
  ratios = p / jnp.where(remaining_probs == 0, 1, remaining_probs)

  keys = split(key, ratios.shape[0])
  remainder, counts = lax_control_flow.scan(f, n, (ratios, keys), unroll=unroll)
  # final remainder should be zero

  return jnp.moveaxis(counts, 0, -1).astype(dtype)