"""
Enhanced caching service with TTL, cleanup, and statistics.
"""

from datetime import datetime, timedelta
from threading import Lock
import time

class CacheEntry:
    __slots__ = ['value', 'expires_at', 'created_at']

    def __init__(self, value, ttl_seconds):
        now = datetime.now()
        self.value = value
        self.created_at = now
        self.expires_at = now + timedelta(seconds=ttl_seconds)

    def is_expired(self):
        return datetime.now() > self.expires_at

    def age_seconds(self):
        return (datetime.now() - self.created_at).total_seconds()


class Cache:
    """Thread-safe in-memory cache with TTL support."""

    # Default TTL values for different data types
    TTL_REALTIME = 60        # 1 minute for realtime quotes
    TTL_KLINE = 1800         # 30 minutes for kline data
    TTL_SIGNAL = 300         # 5 minutes for computed signals
    TTL_FUNDAMENTAL = 3600   # 1 hour for fundamental data
    TTL_STATIC = 86400       # 24 hours for static data (stock names, etc.)

    def __init__(self):
        self._cache = {}
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # Cleanup every 5 minutes

    def get(self, key):
        """Get value from cache if not expired."""
        self._maybe_cleanup()

        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None

            self._hits += 1
            return entry.value

    def set(self, key, value, ttl=None):
        """Set value in cache with TTL."""
        if ttl is None:
            ttl = self.TTL_KLINE

        with self._lock:
            self._cache[key] = CacheEntry(value, ttl)

    def delete(self, key):
        """Delete a specific key."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]

    def clear_pattern(self, pattern):
        """Clear all keys matching a pattern (simple prefix match)."""
        with self._lock:
            keys_to_delete = [k for k in self._cache.keys() if k.startswith(pattern)]
            for key in keys_to_delete:
                del self._cache[key]
            return len(keys_to_delete)

    def clear_all(self):
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def get_stats(self):
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0

            # Count expired entries
            now = datetime.now()
            expired = sum(1 for e in self._cache.values() if e.is_expired())

            return {
                'size': len(self._cache),
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': round(hit_rate, 1),
                'expired_pending': expired
            }

    def get_entry_info(self, key):
        """Get info about a specific cache entry."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            return {
                'exists': True,
                'expired': entry.is_expired(),
                'age_seconds': round(entry.age_seconds(), 1),
                'created_at': entry.created_at.isoformat(),
                'expires_at': entry.expires_at.isoformat()
            }

    def _maybe_cleanup(self):
        """Periodically cleanup expired entries."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        with self._lock:
            self._last_cleanup = now
            keys_to_delete = [k for k, v in self._cache.items() if v.is_expired()]
            for key in keys_to_delete:
                del self._cache[key]


# Global cache instance
cache = Cache()


def cached(key_prefix, ttl=None):
    """Decorator for caching function results."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Build cache key from prefix and arguments
            key_parts = [key_prefix]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)

            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result

            # Call function and cache result
            result = func(*args, **kwargs)
            if result is not None:
                cache.set(cache_key, result, ttl)

            return result
        return wrapper
    return decorator
