# ConfigMaps & Secrets in Kubernetes

## Overview

| Resource | Purpose | Encoding | Encryption at Rest |
|----------|---------|----------|-------------------|
| **ConfigMap** | Non-sensitive configuration | Plain text | Optional |
| **Secret** | Sensitive data | Base64 (not encrypted by default!) | Optional (with etcd encryption) |

> ⚠️ **Important:** Secrets are base64-encoded, NOT encrypted by default! Enable etcd encryption for real security.

## ConfigMap

### Creation Methods

```bash
# From literal values
kubectl create configmap app-config \
  --from-literal=APP_ENV=production \
  --from-literal=LOG_LEVEL=info \
  --from-literal=MAX_CONNECTIONS=100

# From file
kubectl create configmap nginx-config --from-file=nginx.conf

# From env file
kubectl create configmap app-env --from-env-file=app.env

# From directory
kubectl create configmap config-dir --from-file=./config/
```

### YAML Definition

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: default
data:
  # Key-value pairs
  APP_ENV: "production"
  LOG_LEVEL: "info"
  MAX_CONNECTIONS: "100"
  DATABASE_HOST: "postgres.default.svc.cluster.local"
  
  # Multi-line values (file contents)
  app.properties: |
    server.port=8080
    spring.datasource.url=jdbc:postgresql://postgres:5432/mydb
    spring.redis.host=redis
  
  nginx.conf: |
    server {
        listen 80;
        server_name localhost;
        location / {
            proxy_pass http://backend:8080;
        }
    }
```

## Secret

### Types of Secrets

| Type | Use Case |
|------|----------|
| `Opaque` | Generic (default) |
| `kubernetes.io/tls` | TLS certificates |
| `kubernetes.io/dockerconfigjson` | Docker registry auth |
| `kubernetes.io/service-account-token` | Service account tokens |
| `kubernetes.io/basic-auth` | HTTP basic auth |
| `kubernetes.io/ssh-auth` | SSH credentials |

### Creation Methods

```bash
# Generic (Opaque)
kubectl create secret generic db-credentials \
  --from-literal=username=admin \
  --from-literal=password=SuperSecret123!

# TLS certificate
kubectl create secret tls api-tls \
  --cert=server.crt \
  --key=server.key

# Docker registry
kubectl create secret docker-registry registry-creds \
  --docker-server=registry.example.com \
  --docker-username=myuser \
  --docker-password=mypassword \
  --docker-email=myemail@example.com
```

### YAML Definition

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: db-credentials
  namespace: default
type: Opaque
data:
  # Base64 encoded values
  username: YWRtaW4=          # echo -n 'admin' | base64
  password: U3VwZXJTZWNyZXQ=  # echo -n 'SuperSecret' | base64
---
# Alternatively, use stringData (auto-encoded):
apiVersion: v1
kind: Secret
metadata:
  name: db-credentials
type: Opaque
stringData:
  username: admin           # Plain text, auto-encoded
  password: SuperSecret123!
```

## Injection Methods

### Method 1: Environment Variables

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: backend-pod
spec:
  containers:
  - name: backend
    image: backend:latest
    env:
    # Single key from ConfigMap
    - name: APP_ENV
      valueFrom:
        configMapKeyRef:
          name: app-config
          key: APP_ENV
    
    # Single key from Secret
    - name: DB_PASSWORD
      valueFrom:
        secretKeyRef:
          name: db-credentials
          key: password
    
    # All keys from ConfigMap as env vars
    envFrom:
    - configMapRef:
        name: app-config
    
    # All keys from Secret as env vars
    - secretRef:
        name: db-credentials
```

**Pod's environment:**
```
APP_ENV=production
LOG_LEVEL=info
MAX_CONNECTIONS=100
username=admin
password=SuperSecret123!
```

### Method 2: Volume Mounts (Files)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: backend-pod
spec:
  containers:
  - name: backend
    image: backend:latest
    volumeMounts:
    
    # Mount ConfigMap as files
    - name: config
      mountPath: /etc/config
      readOnly: true
    
    # Mount specific file
    - name: nginx-config
      mountPath: /etc/nginx/nginx.conf
      subPath: nginx.conf  # Mount single key as file
    
    # Mount Secrets as files
    - name: secrets
      mountPath: /etc/secrets
      readOnly: true
  
  volumes:
  - name: config
    configMap:
      name: app-config
  
  - name: nginx-config
    configMap:
      name: app-config
  
  - name: secrets
    secret:
      secretName: db-credentials
      defaultMode: 0400  # Restrict permissions
```

**Result in pod:**
```
/etc/config/
    APP_ENV              ← "production"
    LOG_LEVEL            ← "info"
    app.properties       ← multi-line content
    nginx.conf           ← nginx config

/etc/secrets/
    username             ← "admin"
    password             ← "SuperSecret123!"
```

### Method 3: Command Arguments

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: app
    image: myapp:latest
    command: ["/app/start.sh"]
    args:
    - "--env=$(APP_ENV)"
    - "--log-level=$(LOG_LEVEL)"
    env:
    - name: APP_ENV
      valueFrom:
        configMapKeyRef:
          name: app-config
          key: APP_ENV
    - name: LOG_LEVEL
      valueFrom:
        configMapKeyRef:
          name: app-config
          key: LOG_LEVEL
```

## Injection Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ kubectl apply -f app-config.yaml                                │
│   ConfigMap stored in etcd                                      │
└───────────────────┬─────────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────────┐
│ kubectl apply -f backend-pod.yaml                               │
│   Pod spec stored in etcd                                       │
│   Scheduler assigns Pod → Node2                                 │
└───────────────────┬─────────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────────┐
│ kubelet on Node2                                                │
│   1. Receives pod assignment                                    │
│   2. Downloads container image                                  │
│   3. Fetches ConfigMap from API server                         │
│   4. Fetches Secret from API server                            │
│   5. Creates container with:                                    │
│      - Environment variables set                               │
│      - Volume mounts created                                    │
│      - Files written from ConfigMap/Secret data                │
│   6. Starts container                                           │
└───────────────────┬─────────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────────┐
│ Container running                                               │
│   ENV: APP_ENV=production, LOG_LEVEL=info                       │
│   Files: /etc/config/app.properties, /etc/secrets/password      │
└─────────────────────────────────────────────────────────────────┘
```

## Live Updates (Hot Reload)

### Volume-Mounted ConfigMaps Update Automatically

```
ConfigMap updated (kubectl apply or edit)
     │
     ├─→ 1. New version stored in etcd
     │
     ├─→ 2. kubelet detects change (sync period ~1min)
     │
     ├─→ 3. Files on disk updated atomically
     │       (symlink swap, no partial reads)
     │
     ├─→ 4. Application detects file change
     │       (inotify, polling, or signal)
     │
     └─→ 5. Application reloads config
              (app must support hot reload)
```

**Wait time:** ~1-2 minutes (kubelet sync period)

### Environment Variables Do NOT Update

Environment variables are set at container startup only. To update:
- **Delete and recreate** pod (Deployment rolling update)
- Use **volume mounts** if live updates needed

## ConfigMap/Secret in Deployment Rollout

```yaml
# Method: Hash annotation triggers rollout on config change
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
spec:
  template:
    metadata:
      annotations:
        # Checksum changes when ConfigMap changes → triggers rollout
        checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
    spec:
      containers:
      - name: backend
        envFrom:
        - configMapRef:
            name: app-config
```

Or use Helm/Kustomize for automatic rollouts on config change.

## Immutable ConfigMaps/Secrets

**Prevent accidental changes** and improve performance:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: stable-config
immutable: true  # Can only be deleted, not modified
data:
  APP_VERSION: "2.5.0"
  FEATURE_FLAGS: "feature-a,feature-b"
```

**Benefits:**
- Prevents accidental modifications
- kubelet doesn't watch immutable ConfigMaps (reduces API load)
- ~70% reduction in API server load for ConfigMap watches in large clusters

## Docker Registry Secret

```yaml
# Pull from private registry
apiVersion: v1
kind: Pod
metadata:
  name: private-app
spec:
  imagePullSecrets:
  - name: registry-creds
  containers:
  - name: app
    image: registry.example.com/myapp:latest
```

Or attach to ServiceAccount (recommended):
```bash
kubectl patch serviceaccount default \
  -p '{"imagePullSecrets": [{"name": "registry-creds"}]}'
```

## Secret Encryption at Rest

By default, Secrets are stored in etcd as **base64 only** (not encrypted). Enable encryption:

```yaml
# /etc/kubernetes/encryption-config.yaml
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
- resources:
  - secrets
  providers:
  - aescbc:
      keys:
      - name: key1
        secret: <32-byte-base64-key>
  - identity: {}  # Fallback for unencrypted

# Enable in kube-apiserver:
# --encryption-provider-config=/etc/kubernetes/encryption-config.yaml
```

### External Secret Management (Best Practice)

For production, use external secret stores:

**AWS Secrets Manager:**
```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: db-credentials
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: ClusterSecretStore
  target:
    name: db-credentials  # K8s Secret to create
  data:
  - secretKey: password
    remoteRef:
      key: prod/database/credentials
      property: password
```

**HashiCorp Vault:**
```yaml
apiVersion: secrets.hashicorp.com/v1beta1
kind: VaultStaticSecret
metadata:
  name: db-credentials
spec:
  mount: secret
  path: data/production/database
  destination:
    name: db-credentials
    create: true
  refreshAfter: 30s
```

**Sealed Secrets (GitOps-friendly):**
```bash
# Encrypt secret (can be stored in Git safely)
kubeseal --format=yaml < secret.yaml > sealed-secret.yaml

# Sealed controller decrypts and creates Secret
kubectl apply -f sealed-secret.yaml
```

## ConfigMap Size Limits

| Limit | Value |
|-------|-------|
| Max size per ConfigMap | **1 MiB** (1,048,576 bytes) |
| Max size per Secret | **1 MiB** |

For larger data, use object storage (S3, GCS) or ConfigMap projections.

## Best Practices

### Configuration
1. **Separate config from code** - Never hardcode values
2. **Use namespaces** to isolate configs per environment
3. **Label resources** for easy management
4. **Use immutable ConfigMaps** for stable, infrequently changed config

### Security
1. **Never commit Secrets to Git** (use Sealed Secrets, External Secrets, or Vault)
2. **Enable etcd encryption** for Secrets at rest
3. **Use least-privilege RBAC** - Limit who can read Secrets
4. **Rotate credentials regularly** with External Secrets Operator
5. **Use volume mounts** instead of env vars for Secrets (less likely to appear in logs)
6. **Audit Secret access** with audit logging

### Operations
```bash
# View ConfigMap data
kubectl get configmap app-config -o jsonpath='{.data}'

# Decode all Secret values
kubectl get secret db-credentials -o jsonpath='{.data}' | \
  jq 'to_entries | map({key: .key, value: (.value | @base64d)}) | from_entries'

# Check what's using a ConfigMap
kubectl get pods -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.volumes[*].configMap.name}{"\n"}{end}'

# Compare ConfigMap across namespaces
kubectl get configmap app-config -n staging -o yaml
kubectl get configmap app-config -n production -o yaml
```

## Troubleshooting

```bash
# ConfigMap not found in pod
kubectl exec pod-name -- env | grep APP_ENV  # Check env vars
kubectl exec pod-name -- ls /etc/config      # Check mounted files

# Secret not loading
kubectl get secret db-credentials -o yaml     # Verify secret exists and has correct keys

# Config changes not reflected
kubectl exec pod-name -- cat /etc/config/APP_ENV  # For volume mounts, should update
# For env vars: must restart pod

# Permission denied on secret files
kubectl describe pod my-pod  # Check volume mount permissions
```

