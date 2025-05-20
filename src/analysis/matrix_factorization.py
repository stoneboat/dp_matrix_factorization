from typing import Optional, Tuple

import time

import jax
from jax import numpy as jnp
from jax import value_and_grad, jit
from jax import config
import numpy as np

# With large matrices, the extra precision afforded by performing all
# computations in float64 is critical.
config.update('jax_enable_x64', True)


@jit
def diagonalize_and_take_jax_matrix_sqrt(matrix: jnp.ndarray, min_eval: float = 0.0) -> jnp.ndarray:
  """Matrix square root for positive-semi-definite, Hermitian matrices."""
  evals, evecs = jnp.linalg.eigh(matrix)
  eval_sqrt = jnp.maximum(evals, min_eval)**0.5
  sqrt = evecs @ jnp.diag(eval_sqrt) @ evecs.T
  return sqrt

def hermitian_adjoint(matrix: jnp.ndarray) -> jnp.ndarray:
  return jnp.conjugate(matrix).T

def compute_loss_in_x(target: jnp.ndarray, x: jnp.ndarray):
  m = hermitian_adjoint(target) @ target @ jnp.linalg.inv(x)
  raw_trace = jnp.trace(m)
  max_diag = jnp.max(jnp.diag(x))
  return raw_trace * max_diag

def compute_normalized_x_from_vector(matrix_to_factorize, v, precomputed_sqrt: Optional[jnp.ndarray] = None):
  """Computes a normalized (to all-1s diagonal) version of the vector -> matrix
  mapping which defines the relationship between fixed points and optima.

  At a fixed point of phi (equivalently an optimum of the symmetrized
  factorization problem), the normalization below will no-op. But to normalize
  between iterations of the fixed-point method and iterations of the descent-
  based methods, it is useful to force the results of this transformation
  to always have constant-1 diagonals.
  """
  inv_diag_sqrt = jnp.diag(v ** -(0.5))
  diag_sqrt = jnp.diag(v ** 0.5)
  if precomputed_sqrt is None:
    target = hermitian_adjoint(matrix_to_factorize) @ matrix_to_factorize
    matrix_sqrt = diagonalize_and_take_jax_matrix_sqrt(
        diag_sqrt @ target.astype(diag_sqrt.dtype) @ diag_sqrt)
  else:
    # We simply assume that our caller did this computation correctly.
    matrix_sqrt = precomputed_sqrt
  x = inv_diag_sqrt @ matrix_sqrt @ inv_diag_sqrt
  # Force all-1s on diagonal. This normalization is a requirement for the
  # initial iterates of the descent-based methods, and we know it's true at the
  # optimum.
  x_sqrt_diag = jnp.diag(x)
  normalized_x = jnp.diag((x_sqrt_diag)**(-0.5)) @ x @ jnp.diag((x_sqrt_diag)**(-0.5))
  return normalized_x

def optimize_factorization_grad_descent(target: jnp.ndarray, n_iters: int, initial_x: jnp.ndarray, lr: float = 1., use_armijo_rule: bool = True):
  """Uses JAX-implemented gradient descent to optimize DP-MatFac problem."""

  # Capture target in loss definition.
  compute_loss = lambda x: compute_loss_in_x(target=target, x=x)
  compiled_loss = jit(compute_loss)

  def find_next_iterate(x_iter, grad, init_lr):
    candidate = x_iter - grad * init_lr
    non_pd = jnp.any(jnp.isnan(jnp.linalg.cholesky(candidate)))
    if non_pd:
      # We choose 0.1 as the Armijo factor; this is what the paper we're looking to reproduce does as well
      return find_next_iterate(x_iter, grad, init_lr * 0.1)
    else:
      sufficient_decrease_condition = compiled_loss(x_iter) + init_lr * 0.25 * jnp.sum(grad ** 2)
      if compiled_loss(candidate) <= sufficient_decrease_condition:
        return candidate
      return find_next_iterate(x_iter, grad, init_lr * 0.1)

  loss_and_grad = value_and_grad(compute_loss)

  x_iter = initial_x

  loss_array = []
  time_array = []

  start = time.time()
  for i in range(n_iters):
    # Gradient step
    loss, grad = loss_and_grad(x_iter)
    diag_elements = jnp.diag_indices_from(grad)
    grad1 = grad.at[diag_elements].set(0)
    loss_array.append(loss)
    if use_armijo_rule:
      x_iter = find_next_iterate(x_iter, grad1, lr)
    else:
      x_iter = x_iter - lr * grad1
    # Orthogonally project onto symmetric matrices.
    x_iter = (x_iter + x_iter.T) / 2
    
    time_array.append(time.time() - start)
  
  # Suppress any costs to the first iteration
  initial_time = time_array[0]
  time_array = [x - initial_time for x in time_array]
  return x_iter, loss_array, time_array