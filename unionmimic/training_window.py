"""No implicit renewal of an elapsed user-authorized training window."""
import math,time

def remaining_seconds(authorization,now=None):
    if authorization.get('user_authorized') is not True:raise PermissionError('Explicit user authorization is required')
    start=float(authorization['started_unix']);deadline=float(authorization['deadline_unix'])
    now=time.time() if now is None else float(now)
    if not all(math.isfinite(x) for x in (start,deadline,now)) or not 0<deadline-start<=86400.01:
        raise ValueError('Training window must be finite and at most 24 hours')
    if now<start:raise ValueError('Training window has not started')
    if now>=deadline:raise TimeoutError('Authorized training window expired; a new user authorization is required')
    return deadline-now
