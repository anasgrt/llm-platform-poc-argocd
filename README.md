# LLM Platform POC ArgoCD

This repository owns the GitOps layer for the local LLM platform. It contains
both the infrastructure Helm charts (cert-manager, ingress-nginx, Rancher) and
the AI platform workloads, all managed by ArgoCD via the App of Apps pattern.

The infrastructure repository is responsible for:

- Vagrant and VirtualBox VM lifecycle
- k3s control and data nodes
- Namespaces and local TLS secrets (mkcert)
- ArgoCD installation (the only Helm chart managed outside GitOps)
- Applying the root App of Apps Application

This repository is responsible for:

- **Platform infrastructure** (via App of Apps sync waves):
  - cert-manager (wave 0)
  - ingress-nginx (wave 1)
  - Rancher (wave 2)
- **`ai-platform` workloads** (wave 3): Qdrant, Qwen3 server, embedding server,
  RAG app, Fluent Bit, log retention, and sample-log ingestion
- **`monitoring` workloads** (wave 3): Prometheus, Grafana, node-exporter, and
  kube-state-metrics
- ArgoCD `Application` manifests for dev and prod
- Image build and dev-overlay image tag updates through GitHub Actions

## Layout

```txt
argocd/
  root.yaml                    # Root App of Apps (applied by setup.sh)
  app-prod.yaml                # Standalone prod Application (manual sync)
deploy/
  platform/                    # App of Apps child Applications
    cert-manager.yaml           # sync wave 0
    ingress-nginx.yaml          # sync wave 1
    rancher.yaml                # sync wave 2
    workloads-dev.yaml          # sync wave 3
  base/
    *.yaml
    sample-logs/
  overlays/
    dev/
    prod/
images/
  embedding-server/
  ingestion/
  qwen3-server/
  rag-app/
```

## Bootstrap

First bring up the infrastructure repo:

```bash
cd ../llm-platform-poc
vagrant up
```

The infrastructure Vagrant bootstrap installs ArgoCD and applies the root App of
Apps automatically. The root Application points at `deploy/platform/` in this
repository, which contains child Applications that install the full stack via
sync waves.

If you need to recreate the root Application manually from the control node:

```bash
vagrant ssh control -c \
  'kubectl apply -f https://raw.githubusercontent.com/anasgrt/LLM-PLATFORM-POC-ARGOCD/main/argocd/root.yaml'
```

Production workloads use a separate Application with manual sync via
`argocd/app-prod.yaml`.

## Validate Locally

Render the dev overlay before pushing:

```bash
kubectl kustomize deploy/overlays/dev >/tmp/llm-platform-dev.yaml
```

Render the prod overlay:

```bash
kubectl kustomize deploy/overlays/prod >/tmp/llm-platform-prod.yaml
```

## Images

The workflow in `.github/workflows/build-and-deploy.yaml` builds these images
and pushes them to GHCR:

- `ghcr.io/anasgrt/qwen3-server`
- `ghcr.io/anasgrt/embedding-server`
- `ghcr.io/anasgrt/rag-app`
- `ghcr.io/anasgrt/ingestion`

The workflow bumps `deploy/overlays/dev/kustomization.yaml` to the pushed Git
SHA. Promote to prod by copying known-good dev tags into
`deploy/overlays/prod/kustomization.yaml`.

## Sample Ingestion

`deploy/base/kustomization.yaml` generates the `sample-logs` ConfigMap from
`deploy/base/sample-logs/*.log`. `05-ingestion-job.yaml` is an ArgoCD PostSync
hook, so it runs after the API, embedding server, and Qdrant manifests sync.

## Access

After ArgoCD completes all sync waves:

| What | URL |
|------|-----|
| ArgoCD | `https://argocd.localhost:8443` |
| Rancher | `https://rancher.localhost:8443` |
| Chat UI | `https://chat.localhost:8443` |
| Grafana | `https://grafana.localhost:8443` |
| Prometheus | `https://prometheus.localhost:8443` |

Test the API:

```bash
curl -k -X POST https://chat.localhost:8443/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"question": "What errors keep recurring?"}'
```

Monitor sync progress:

```bash
kubectl get applications -n argocd
```
