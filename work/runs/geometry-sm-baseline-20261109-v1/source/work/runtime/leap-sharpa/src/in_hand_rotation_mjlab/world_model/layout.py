from __future__ import annotations

def state_slices(nj: int) -> dict[str, slice]:
  names = (("q", nj), ("qd", nj), ("cmd", nj), ("p", 3), ("quat", 4), ("v", 3), ("w", 3))
  out, i = {}, 0
  for n, d in names:
    out[n] = slice(i, i + d)
    i += d
  return out
