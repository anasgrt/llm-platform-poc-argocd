# LLM Platform POC ArgoCD

This repository owns the GitOps layer for the local LLM platform. It contains
the Kubernetes workloads that ArgoCD syncs into the k3s cluster created by the
Vagrant/Rancher repository.

The infrastructure repository is responsible for:

- Vagrant and VirtualBox VM lifecycle
- k3s control and data nodes
- cert-manager, ingress-nginx, Rancher, and ArgoCD installation
- local TLS secrets and hostPath model storage

This repository is responsible for:

- `ai-platform` workloads: Qdrant, Qwen3 server, embedding server, RAG app,
  Fluent Bit, log retention, and sample-log ingestion
- `monitoring` workloads: Prometheus, Grafana, node-exporter, and
  kube-state-metrics
- ArgoCD `Application` manifests for dev and prod
- image build and dev-overlay image tag updates through GitHub Actions

## Layout

```txt
argocd/
  app-dev.yaml
  app-prod.yaml
deploy/
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

The infrastructure Vagrant bootstrap creates the dev ArgoCD `Application`
automatically after ArgoCD is installed. That Application points at this
repository and syncs `deploy/overlays/dev`.

If you need to recreate it manually from the control node, use:

```bash
vagrant ssh control -c \
  'kubectl apply -f https://raw.githubusercontent.com/anasgrt/LLM-PLATFORM-POC-ARGOCD/main/argocd/app-dev.yaml'
```

Production is intentionally manual-sync through `argocd/app-prod.yaml`.

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

After ArgoCD syncs successfully:

| What | URL |
|------|-----|
| Chat UI | `https://chat.localhost:8443` |
| Grafana | `https://grafana.localhost:8443` |
| Prometheus | `https://prometheus.localhost:8443` |
| ArgoCD | `https://argocd.localhost:8443` |
| Rancher | `https://rancher.localhost:8443` |

Test the API:

```bash
curl -k -X POST https://chat.localhost:8443/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"question": "What errors keep recurring?"}'
```
