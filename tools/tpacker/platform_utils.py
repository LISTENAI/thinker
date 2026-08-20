def normalize_platform_name(platform):
    """Return the canonical platform string used by graph-analysis code."""
    if platform is None:
        return None
    if isinstance(platform, bytes):
        platform = platform.decode()
    name = str(platform)
    if name.upper() == "VENUSA":
        return "venusA"
    if name.upper() == "VENUS":
        return "venus"
    if name.upper() == "ARCS":
        return "arcs"
    if name.upper() == "MARS":
        return "mars"
    return name


__all__ = ["normalize_platform_name"]
