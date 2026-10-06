# GCP production deployment

This deployment path closes the infrastructure-as-code gap between the local
Compose stack and a production-style platform. It provisions a regional GKE
cluster and Artifact Registry; Neo4j, RabbitMQ, Redis, and TimescaleDB are
intentionally external managed services. A Kubernetes cluster is not an
appropriate substitute for Neo4j Enterprise clustering or managed backups.

## Prerequisites

- Terraform 1.7+, Google Cloud SDK, `kubectl`, and `gke-gcloud-auth-plugin`.
- A GCP project and permission to create GKE, Artifact Registry, service
  accounts, and enable APIs.
- Production endpoints and credentials for Neo4j, RabbitMQ, Redis, TimescaleDB
  and the configured LLM providers. Keep them in Secret Manager or an External
  Secrets controller; never commit them to this repository.

## Provision the platform foundation

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit project_id and capacity values.
terraform init
terraform plan
terraform apply

gcloud container clusters get-credentials graphrag-prod \
  --region europe-west3 --project YOUR_PROJECT_ID
```

Run the local infrastructure checks before applying:

```bash
make terraform-fmt
make terraform-validate
make terraform-test
checkov -d infra/terraform --framework terraform
```

`terraform test` uses a mocked Google provider and checks platform invariants
without creating GCP resources. Checkov is a static security scan and does not
need cloud credentials.

## Local Kubernetes validation

Validate the rendered objects locally with Docker Desktop Kubernetes, kind, or
k3d before applying them to GKE. These commands verify deployability and
intended controls; they do not establish production capacity or uptime.

```powershell
kubectl kustomize deploy/kubernetes > artifacts/kubernetes-rendered.yaml
kubectl apply --dry-run=client -k deploy/kubernetes
kubectl auth can-i --list --as=system:serviceaccount:graphrag:graphrag-workload -n graphrag
```

With a local cluster and locally tagged images, exercise rollout, readiness,
restart, and rollback:

```powershell
kubectl apply -k deploy/kubernetes
kubectl rollout status deployment/graphrag-api -n graphrag
kubectl rollout status deployment/graphrag-mcp -n graphrag
kubectl rollout restart deployment/graphrag-mcp -n graphrag
kubectl rollout undo deployment/graphrag-mcp -n graphrag
kubectl get networkpolicy,pdb,hpa -n graphrag
```

Record cluster version, node capacity, image digests, pod counts, rollout time,
restart recovery, and probe failures in the evidence report.

Build and push one image per runtime service. The Dockerfile already supports
the service-specific dependency sets through `--build-arg SERVICE=...`.

```bash
docker build --build-arg SERVICE=api -t REGION-docker.pkg.dev/PROJECT_ID/graphrag/api:TAG .
docker build --build-arg SERVICE=ingestion -t REGION-docker.pkg.dev/PROJECT_ID/graphrag/ingestion:TAG .
docker build --build-arg SERVICE=query -t REGION-docker.pkg.dev/PROJECT_ID/graphrag/query:TAG .
docker build --build-arg SERVICE=backup -t REGION-docker.pkg.dev/PROJECT_ID/graphrag/backup:TAG .
docker push REGION-docker.pkg.dev/PROJECT_ID/graphrag/api:TAG
docker push REGION-docker.pkg.dev/PROJECT_ID/graphrag/ingestion:TAG
docker push REGION-docker.pkg.dev/PROJECT_ID/graphrag/query:TAG
docker push REGION-docker.pkg.dev/PROJECT_ID/graphrag/backup:TAG
```

## Deploy the workloads

Copy `deploy/kubernetes/secrets.example.yaml` outside the repository, insert
base64-encoded values (or apply an equivalent ExternalSecret), and replace each
`REGION`, `PROJECT_ID`, and `TAG` image placeholder in the manifests. Also replace
`PROJECT_ID` in `service-account.yaml`; Terraform binds this Kubernetes service
account to the GCP workload identity.

Set `ENERGY_GOVERNANCE_DB_URL` in the secret to a shared async PostgreSQL URL
(`postgresql+asyncpg://...`). Kubernetes will refuse to start the API pod if
this required secret key is missing. Published RDF blobs use `/data/energy/published` on the
`energy-published-rdf` claim. Before applying Kustomize, provision a storage
class that supports `ReadWriteMany` across API replicas and set
`storageClassName` in `energy-published-rdf-pvc.yaml` when your cluster has no
default RWX class. Back up both the database and the blob volume together;
the database stores blob paths and publication pointers.

In `deploy/kubernetes/configmap.yaml`, also replace the `REPLACE_ME` values for
`GRAPHRAG_API_RESOURCE`, `GRAPHRAG_MCP_RESOURCE` and `CORS_ORIGINS`. Every pod
runs with `ENV=production`, where settings validation rejects a non-HTTPS or
missing `CORS_ORIGINS`, so leaving it unset crash-loops the API, MCP and worker
pods at startup.

```bash
kubectl apply -f /secure/path/graphrag-secrets.yaml
kubectl apply -k deploy/kubernetes
kubectl -n graphrag rollout status deployment/graphrag-api
kubectl -n graphrag rollout status deployment/graphrag-ingestion-worker
kubectl -n graphrag rollout status deployment/graphrag-query-worker
```

The API is internally exposed by default. Add a managed ingress or API gateway
with TLS, an approved hostname, and an explicit WAF/rate-limit policy before
opening it to clients.

## Operational acceptance checks

1. Verify `GET /health/ready` checks Neo4j, Redis and provider health.
2. Run a controlled ingestion, query it, and verify worker results flow through Redis.
3. Verify Prometheus scrapes `/metrics` and OTEL exports reach the configured
   `OTEL_EXPORTER_OTLP_ENDPOINT`.
4. Run the backup/restore exercise from `docs/runbook.md` against a non-production
   tenant and record the measured RPO/RTO.
5. Configure queue-depth-based autoscaling (KEDA or equivalent) before relying
   on ingestion throughput; the supplied HPA covers API CPU only.

`deploy/kubernetes/keda-scaledobject.example.yaml` provides queue-aware
scaling definitions for the existing ingestion and query queues. It is not part
of the default Kustomize set because it requires the KEDA CRDs to be installed
and RabbitMQ management access approved for the scaler.

Terraform also provisions a versioned GCS bucket with lifecycle retention and
Workload Identity access for the application account. After a manual backup and
restore drill, apply `deploy/kubernetes/backup-cronjob.example.yaml`, replacing
its image placeholders and `GRAPHRAG_BACKUP_BUCKET` with the `backup_bucket_uri`
output without the `gs://` prefix. The CronJob is deliberately opt-in because
tenant scope and recovery acceptance must be approved by the data owner.

## Explicit production boundaries

- This repository does not provision Neo4j Enterprise/Aura, its read replicas,
  clustering, or online backups. Select and operate that service according to
  the required RPO/RTO and tenant-isolation model.
- Terraform creates the compute/control-plane foundation, not DNS, WAF,
  Secret Manager entries, or third-party data stores; those need organisation
  specific ownership and credentials.
- No capacity or SLA claim should be made until a representative load test and
  recovery drill have been recorded for the selected managed services.
- The query worker receives an ephemeral Hugging Face cache because it warms a
  reranker on startup. For predictable startup time and egress control, bake
  the approved model into the query image before operating at scale.
