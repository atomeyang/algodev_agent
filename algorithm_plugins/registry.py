"""Business-plugin registry: resolve a task *domain* to its plugin object.

Guide 4.1: a task references one canonical business plugin (by domain) instead
of copying domain code into every task. Plugins live under
``algorithm_plugins/<name>/plugin.py`` and expose either a module-level
``PLUGIN`` instance or a class whose instance carries a ``domain`` attribute.

Only the *base* protocol in :mod:`algorithm_plugins.common.plugin` is required
(validate_requirement / validate_dataset / default_slices / deployment_scope).
Everything else is an optional hook; the graph nodes fall back to the built-in
synthetic-tabular behaviour when a hook is absent, so the existing demo task
keeps its exact semantics.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PLUGINS_DIR = Path(__file__).resolve().parent
DEFAULT_DOMAIN = "demo_tabular"

_cache: Dict[str, Any] = {}


def _ensure_importable() -> None:
    """Make ``algorithm_plugins`` importable as a package.

    The platform is often run straight from a checkout (``PYTHONPATH=...``) and
    plugin modules may import sibling modules (``algorithm_plugins.iqa.metrics``);
    putting the repo root on ``sys.path`` once keeps that working without the
    caller having to know.
    """
    root = str(PLUGINS_DIR.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


def available_domains() -> List[str]:
    if not PLUGINS_DIR.exists():
        return []
    skip = {"common", "registry"}
    return sorted(d.name for d in PLUGINS_DIR.iterdir()
                  if d.is_dir() and (d / "plugin.py").exists()
                  and not d.name.startswith(("_", ".")) and d.name not in skip)


def load_plugin(domain: str) -> Optional[Any]:
    """Load and cache the plugin object for *domain* (``None`` when absent)."""
    domain = (domain or DEFAULT_DOMAIN).strip() or DEFAULT_DOMAIN
    if domain in _cache:
        return _cache[domain]
    _ensure_importable()
    module = None
    try:
        module = importlib.import_module(f"algorithm_plugins.{domain}.plugin")
    except Exception:
        module = None
    plugin = _instance_from_module(module, domain) if module is not None else None
    _cache[domain] = plugin
    return plugin


def _instance_from_module(module: Any, domain: str) -> Optional[Any]:
    plugin = getattr(module, "PLUGIN", None)
    if plugin is not None:
        return plugin
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and getattr(obj, "domain", None) == domain:
            return obj()
    # a class may declare a different alias for the same domain folder
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and hasattr(obj, "validate_requirement") and hasattr(obj, "deployment_scope"):
            return obj()
    return None


def resolve_domain(spec: Any) -> str:
    """Domain for a TaskSpec-like object (falls back to the demo domain)."""
    domain = getattr(spec, "domain", "") or ""
    if not domain and isinstance(spec, dict):
        domain = spec.get("domain", "") or ""
    return (domain or DEFAULT_DOMAIN).strip() or DEFAULT_DOMAIN


def resolve_plugin(spec_or_domain: Any) -> Optional[Any]:
    """Plugin for a TaskSpec (or a raw domain string)."""
    if isinstance(spec_or_domain, str):
        return load_plugin(spec_or_domain)
    return load_plugin(resolve_domain(spec_or_domain))


def detect_domain(text: str) -> str:
    """Pick a domain for a natural-language requirement.

    A plugin may opt in by exposing ``matches(text) -> bool``; the first match
    (in directory order) wins, otherwise we keep the demo domain so existing
    behaviour is untouched.
    """
    lowered = (text or "").lower()
    for domain in available_domains():
        plugin = load_plugin(domain)
        matcher = getattr(plugin, "matches", None)
        if callable(matcher):
            try:
                if matcher(text):
                    return domain
            except Exception:
                continue
    return DEFAULT_DOMAIN


def hook(plugin: Any, name: str, default: Any = None) -> Any:
    """Return a plugin hook when the plugin opts in, else *default*."""
    fn = getattr(plugin, name, None)
    return fn if callable(fn) else default


__all__ = ["available_domains", "load_plugin", "resolve_domain", "resolve_plugin",
           "detect_domain", "hook", "DEFAULT_DOMAIN", "PLUGINS_DIR"]
