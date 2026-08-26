group "default" {
  targets = ["base", "agy", "claude", "codex", "cursor", "opencode"]
}

# Tagged distinctly from the `:latest` every backend Dockerfile's `ARG
# BASE_IMAGE` defaults to. Buildx's named-context override (below) that
# repoints a backend's `FROM ${BASE_IMAGE}` at this target's own build
# result -- instead of a registry pull -- only takes effect for a
# non-`:latest` reference; passed as `BASE_IMAGE` build-arg per backend, it
# never changes what a plain, un-overridden build defaults to.
target "base" {
  context    = "."
  dockerfile = "docker/base.Dockerfile"
  tags       = ["helix-runner-base:ci-build"]
  cache-from = ["type=gha,scope=fixture-base"]
  cache-to   = ["type=gha,mode=max,scope=fixture-base"]
}

# Every backend Dockerfile does `FROM ${BASE_IMAGE}`. Pointing that
# reference at the `base` target's own build result (rather than a registry
# pull or the host daemon's image store) lets one `docker buildx bake`
# build the whole fixture set -- base once, then every backend in parallel
# -- without ever pushing an intermediate image anywhere.
target "backend" {
  args = {
    BASE_IMAGE = "helix-runner-base:ci-build"
  }
  contexts = {
    "helix-runner-base:ci-build" = "target:base"
  }
}

target "agy" {
  inherits   = ["backend"]
  context    = "."
  dockerfile = "docker/agy.Dockerfile"
  tags       = ["ghcr.io/ke7/helix-evo-runner-agy:latest"]
  cache-from = ["type=gha,scope=fixture-agy"]
  cache-to   = ["type=gha,mode=max,scope=fixture-agy"]
}

target "claude" {
  inherits   = ["backend"]
  context    = "."
  dockerfile = "docker/claude.Dockerfile"
  tags       = ["ghcr.io/ke7/helix-evo-runner-claude:latest"]
  cache-from = ["type=gha,scope=fixture-claude"]
  cache-to   = ["type=gha,mode=max,scope=fixture-claude"]
}

target "codex" {
  inherits   = ["backend"]
  context    = "."
  dockerfile = "docker/codex.Dockerfile"
  tags       = ["ghcr.io/ke7/helix-evo-runner-codex:latest"]
  cache-from = ["type=gha,scope=fixture-codex"]
  cache-to   = ["type=gha,mode=max,scope=fixture-codex"]
}

target "cursor" {
  inherits   = ["backend"]
  context    = "."
  dockerfile = "docker/cursor.Dockerfile"
  tags       = ["ghcr.io/ke7/helix-evo-runner-cursor:latest"]
  cache-from = ["type=gha,scope=fixture-cursor"]
  cache-to   = ["type=gha,mode=max,scope=fixture-cursor"]
}

target "opencode" {
  inherits   = ["backend"]
  context    = "."
  dockerfile = "docker/opencode.Dockerfile"
  tags       = ["ghcr.io/ke7/helix-evo-runner-opencode:latest"]
  cache-from = ["type=gha,scope=fixture-opencode"]
  cache-to   = ["type=gha,mode=max,scope=fixture-opencode"]
}
