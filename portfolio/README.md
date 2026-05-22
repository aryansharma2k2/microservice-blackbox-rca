# Portfolio Dashboard

This directory is a dependency-free static dashboard for the microservice RCA project.
It is designed for free hosting on GitHub Pages or Cloudflare Pages.

## Local Preview

```bash
python3 -m http.server 4173 --directory portfolio
```

Open `http://localhost:4173`.

## Deploy On GitHub Pages

The repository includes `.github/workflows/pages.yml`. In GitHub:

1. Go to `Settings -> Pages`.
2. Set `Source` to `GitHub Actions`.
3. Push to `main`.

The workflow uploads this `portfolio/` directory as the static site.

## Why Static?

The real experiment uses Kubernetes, Prometheus, and Chaos Mesh. Public free
hosting cannot run privileged Kubernetes fault injection safely, so the hosted
portfolio presents a replayable experiment artifact. The full system remains
reproducible locally through the repo's existing `infra/` and `eval/` scripts.
