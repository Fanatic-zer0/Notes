# YAML and the Kubernetes REST API

## Overview

Every Kubernetes resource is defined in YAML and managed through a REST API. Understanding YAML syntax and the Kubernetes API is essential for writing manifests, debugging issues, and building controllers or tooling.

## 1. YAML Fundamentals

### What is YAML?

YAML (YAML Ain't Markup Language) is a human-readable data serialization format. It is a superset of JSON.

### YAML Syntax Rules

**Indentation:** Uses spaces (NOT tabs). 2 spaces is convention.

```yaml
# Comments start with #

# Key-value pair
name: my-app
version: 1.0

# Nested objects (use indentation)
metadata:
  name: my-pod
  namespace: default
  labels:
    app: my-app
    tier: frontend

# Lists with dashes
containers:
  - name: nginx
    image: nginx:1.25
  - name: sidecar
    image: busybox

# Inline list (JSON-style)
args: ["--flag1", "--flag2"]

# Multiline strings
command: |
  #!/bin/sh
  echo "hello"
  exec nginx -g "daemon off;"

# Folded multiline (newlines become spaces)
description: >
  This is a very long description
  that wraps across lines but is
  treated as a single line.
```

### YAML Data Types

```yaml
# String (can omit quotes unless contains special chars)
name: my-app
image: "nginx:1.25"      # quotes when colon in value
path: "/usr/local/bin"

# Integer
replicas: 3
port: 8080

# Float
cpu: 0.5

# Boolean
enabled: true
debug: false

# Null
externalIP: null  # or ~

# Multi-line string (literal block - preserves newlines)
script: |
  echo line 1
  echo line 2

# Multi-line string (folded - newlines become spaces)
message: >
  This becomes one line
  in the output

# Array
ports:
  - 80
  - 443

# Object
resources:
  requests:
    cpu: "100m"
    memory: "128Mi"
  limits:
    cpu: "500m"
    memory: "256Mi"
```

### Common YAML Pitfalls

```yaml
# PITFALL 1: Tab vs Space
spec:
	containers:    # ERROR: tab character!
  containers:    # OK: spaces

# PITFALL 2: Unquoted special characters
value: foo: bar   # ERROR: unexpected colon
value: "foo: bar" # OK

# PITFALL 3: Numbers vs strings
version: 1.0   # Parsed as FLOAT (1.0)
version: "1.0" # Parsed as STRING "1.0"

# Kubernetes often requires strings:
imagePullPolicy: IfNotPresent  # OK - no special chars

# PITFALL 4: Implicit type conversion
country: NO        # YAML parses as boolean false!
country: "NO"      # String "NO"

# PITFALL 5: Anchor & alias (useful but confusing)
defaults: &defaults
  replicas: 3
  image: nginx

deployment1:
  <<: *defaults     # Merge defaults
  name: deploy1

deployment2:
  <<: *defaults     # Merge defaults
  name: deploy2
  replicas: 5       # Override
```

### Validate YAML

```bash
# Validate YAML syntax
kubectl apply --dry-run=client -f manifest.yaml

# Check what would be applied
kubectl apply --dry-run=server -f manifest.yaml

# Pretty print
kubectl apply -f manifest.yaml -o yaml

# Convert YAML to JSON (debug)
cat manifest.yaml | python3 -c "import sys, yaml, json; print(json.dumps(yaml.safe_load(sys.stdin), indent=2))"
```

## 2. Kubernetes API Concepts

### The API is RESTful

Every Kubernetes resource maps to a REST endpoint:

```
GET    /api/v1/namespaces/default/pods           → List pods
POST   /api/v1/namespaces/default/pods           → Create pod
GET    /api/v1/namespaces/default/pods/my-pod    → Get specific pod
PUT    /api/v1/namespaces/default/pods/my-pod    → Replace pod
PATCH  /api/v1/namespaces/default/pods/my-pod    → Update pod
DELETE /api/v1/namespaces/default/pods/my-pod    → Delete pod
```

### kubectl is Just an API Client

```bash
# See the actual API calls kubectl makes
kubectl get pods -v=8

# OUTPUT (truncated):
# GET https://kubernetes.default:6443/api/v1/namespaces/default/pods
# Request Headers:
#   Accept: application/json;as=Table...
#   Authorization: Bearer <token>
# Response Status: 200 OK
```

### API Groups

Kubernetes organizes resources into API groups:

```
Core API (/api/v1):
  Pods, Services, ConfigMaps, Secrets, Namespaces, Nodes, PVs, PVCs

Named API Groups (/apis/GROUP/VERSION):
  apps/v1:             Deployments, ReplicaSets, StatefulSets, DaemonSets
  batch/v1:            Jobs, CronJobs
  networking.k8s.io:   Ingresses, NetworkPolicies
  rbac.authorization:  Roles, ClusterRoles, Bindings
  storage.k8s.io:      StorageClasses
  autoscaling:         HorizontalPodAutoscalers
  apiextensions.k8s.io: CustomResourceDefinitions (CRDs)
```

**In YAML:**
```yaml
# Core API (no group)
apiVersion: v1
kind: Pod

# Named group
apiVersion: apps/v1
kind: Deployment

# Full API path: /apis/apps/v1/namespaces/{ns}/deployments
```

### Discover API Resources

```bash
# List all API resources
kubectl api-resources

# OUTPUT:
# NAME              SHORTNAMES  APIVERSION  NAMESPACED  KIND
# pods              po          v1          true        Pod
# deployments       deploy      apps/v1     true        Deployment
# services          svc         v1          true        Service
# nodes             no          v1          false       Node

# List API versions
kubectl api-versions

# Explain a resource (built-in docs!)
kubectl explain pod
kubectl explain pod.spec
kubectl explain pod.spec.containers
kubectl explain pod.spec.containers.resources
```

## 3. Resource Structure

### Every Kubernetes Resource Has 4 Parts

```yaml
# 1. TypeMeta - What kind of resource is this?
apiVersion: apps/v1
kind: Deployment

# 2. ObjectMeta - Identity and metadata
metadata:
  name: my-deployment
  namespace: default
  labels:
    app: my-app
    version: v1.0
  annotations:
    deployment.kubernetes.io/revision: "1"
    description: "Main application"
  uid: 123e4567-e89b-12d3-a456-426614174000
  resourceVersion: "12345"
  generation: 3
  creationTimestamp: "2024-01-15T10:00:00Z"
  ownerReferences:          # For garbage collection
  - apiVersion: apps/v1
    kind: ReplicaSet
    name: my-replicaset
    uid: abc-def-...

# 3. Spec - Desired state (YOU define this)
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: app
        image: nginx:1.25

# 4. Status - Current state (Kubernetes fills this in)
status:
  availableReplicas: 3
  readyReplicas: 3
  replicas: 3
  updatedReplicas: 3
  conditions:
  - type: Available
    status: "True"
    lastUpdateTime: "2024-01-15T10:01:30Z"
```

### Labels vs Annotations

```yaml
metadata:
  labels:
    # Selectable! Used in selectors, networkpolicies
    app: nginx
    tier: frontend
    env: production
    version: "1.25"
    
  annotations:
    # NOT selectable. Free-form data for tools/humans
    kubernetes.io/ingress.class: "nginx"
    deployment.kubernetes.io/revision: "3"
    prometheus.io/scrape: "true"
    prometheus.io/port: "9090"
    team: "platform-engineering"
    docs: "https://wiki.example.com/my-app"
```

## 4. Interacting with the API Directly

### kubectl proxy

```bash
# Start proxy (handles authentication for you)
kubectl proxy --port=8001 &

# Now use curl without auth headers
curl http://localhost:8001/api/v1/namespaces/default/pods | jq .

curl http://localhost:8001/apis/apps/v1/namespaces/default/deployments | jq .items[].metadata.name
```

### Direct API Access with curl

```bash
# Get service account token
TOKEN=$(kubectl create token default)

# Get API server endpoint
APISERVER=$(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}')

# Get CA certificate
kubectl config view --minify --raw -o jsonpath='{.clusters[0].cluster.certificate-authority-data}' | base64 -d > /tmp/ca.crt

# Call API directly
curl --cacert /tmp/ca.crt \
  -H "Authorization: Bearer $TOKEN" \
  $APISERVER/api/v1/namespaces/default/pods
```

### From Inside a Pod

```bash
# Service account credentials are auto-mounted
ls /var/run/secrets/kubernetes.io/serviceaccount/
# ca.crt  namespace  token

# Call API from pod
TOKEN=$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)
curl --cacert /var/run/secrets/kubernetes.io/serviceaccount/ca.crt \
  -H "Authorization: Bearer $TOKEN" \
  https://kubernetes.default.svc/api/v1/namespaces/default/pods
```

## 5. kubectl Output Formats

```bash
# Default table format
kubectl get pods

# YAML output (full resource)
kubectl get pod my-pod -o yaml

# JSON output
kubectl get pod my-pod -o json

# JSONPath - extract specific fields
kubectl get pod my-pod -o jsonpath='{.status.podIP}'
kubectl get pods -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.podIP}{"\n"}{end}'

# Custom columns
kubectl get pods -o custom-columns='NAME:.metadata.name,IP:.status.podIP,NODE:.spec.nodeName'

# Wide (adds node info)
kubectl get pods -o wide

# Sort by field
kubectl get pods --sort-by=.metadata.creationTimestamp
```

## 6. Patches and Updates

### Three Patch Types

```bash
# Strategic Merge Patch (default, Kubernetes-aware)
kubectl patch deployment my-deploy -p '{"spec":{"replicas":5}}'

# JSON Merge Patch
kubectl patch deployment my-deploy \
  --type merge \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"app","image":"nginx:1.26"}]}}}}'

# JSON Patch (RFC 6902 - precise)
kubectl patch deployment my-deploy \
  --type json \
  -p '[{"op":"replace","path":"/spec/replicas","value":5}]'
```

### Apply vs Create

```bash
# Create - fails if exists
kubectl create -f manifest.yaml

# Apply - create or update (uses 3-way merge)
kubectl apply -f manifest.yaml
# Stores "last applied configuration" in annotation
kubectl.kubernetes.io/last-applied-configuration

# Replace - delete and recreate
kubectl replace -f manifest.yaml
```

## 7. Resource Versioning

### resourceVersion and Optimistic Locking

```yaml
metadata:
  resourceVersion: "12345"  # Set by API server, changes on every update
```

**Why it matters:**
```bash
# Get pod
kubectl get pod my-pod -o yaml > pod.yaml

# Someone else updates the pod (resourceVersion changes to 12346)

# Your update attempt:
kubectl apply -f pod.yaml
# Error: the object has been modified; please apply your changes to the latest version
# (Your "12345" ≠ current "12346" → conflict!)

# Solution: fetch latest first
kubectl get pod my-pod -o yaml > pod.yaml  # Gets 12346
kubectl apply -f pod.yaml                  # Now succeeds
```

### generation and observedGeneration

```yaml
metadata:
  generation: 5          # Incremented each spec change

status:
  observedGeneration: 5  # What controller has processed
  # If generation != observedGeneration → rollout in progress!
```

## 8. Field Selectors and Label Selectors

### Label Selectors

```bash
# Equality
kubectl get pods -l app=nginx
kubectl get pods -l app=nginx,env=production

# Set-based
kubectl get pods -l 'env in (staging, production)'
kubectl get pods -l 'app notin (redis, mysql)'
kubectl get pods -l '!deprecated'  # Has no label 'deprecated'
```

### Field Selectors

```bash
# Filter by resource fields
kubectl get pods --field-selector status.phase=Running
kubectl get pods --field-selector spec.nodeName=node1
kubectl get events --field-selector type=Warning
```

## 9. Watch and Streaming

### Watch Resources

```bash
# Watch pod changes
kubectl get pods -w
# Shows: ADDED, MODIFIED, DELETED events

# Via API
curl http://localhost:8001/api/v1/namespaces/default/pods?watch=true
# Returns: newline-delimited JSON stream
# {"type":"ADDED","object":{...}}
# {"type":"MODIFIED","object":{...}}
```

### Informers (Controller Pattern)

```go
// This is how controllers work internally
// 1. List all existing resources
// 2. Watch for changes
// 3. Process events through work queue

informer := cache.NewInformer(
    listWatch,
    &v1.Pod{},
    resyncPeriod,
    cache.ResourceEventHandlerFuncs{
        AddFunc:    func(obj interface{}) { /* handle new pod */ },
        UpdateFunc: func(old, new interface{}) { /* handle update */ },
        DeleteFunc: func(obj interface{}) { /* handle delete */ },
    },
)
```

## 10. YAML Best Practices for Kubernetes

### Always Set

```yaml
metadata:
  labels:
    app: my-app     # Required for service selectors
    version: v1.0   # For canary deployments

spec:
  containers:
  - resources:       # ALWAYS set
      requests:
        cpu: "100m"
        memory: "128Mi"
      limits:
        cpu: "500m"
        memory: "256Mi"
    
    readinessProbe:  # ALWAYS set (traffic gating)
      httpGet:
        path: /healthz
        port: 8080
      initialDelaySeconds: 10
      periodSeconds: 5

    livenessProbe:   # Set carefully (avoid premature restarts)
      httpGet:
        path: /healthz
        port: 8080
      initialDelaySeconds: 30
      periodSeconds: 10
```

### Namespace Best Practices

```yaml
# Always specify namespace explicitly
metadata:
  namespace: my-team  # Don't rely on default

# Use namespace in kubectl commands
kubectl get pods -n my-team
kubectl apply -f manifest.yaml -n my-team
```

### Linting and Validation

```bash
# Validate locally
kubectl apply --dry-run=client -f manifest.yaml

# Validate against running cluster
kubectl apply --dry-run=server -f manifest.yaml

# Use kubeval for offline validation
kubeval manifest.yaml

# Use kube-score for best practice checks
kube-score score manifest.yaml
```

## 11. Reference: Common Manifest Patterns

### Complete Pod Spec

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: complete-pod
  namespace: default
  labels:
    app: my-app
  annotations:
    prometheus.io/scrape: "true"
spec:
  serviceAccountName: my-service-account
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 2000
  initContainers:
  - name: init-db
    image: busybox
    command: ["sh", "-c", "until nc -z db-service 5432; do sleep 1; done"]
  containers:
  - name: app
    image: my-app:v1.0
    ports:
    - containerPort: 8080
      protocol: TCP
    env:
    - name: DB_HOST
      value: "db-service"
    - name: DB_PASSWORD
      valueFrom:
        secretKeyRef:
          name: db-secret
          key: password
    resources:
      requests:
        cpu: "100m"
        memory: "128Mi"
      limits:
        cpu: "500m"
        memory: "256Mi"
    readinessProbe:
      httpGet:
        path: /ready
        port: 8080
      initialDelaySeconds: 5
      periodSeconds: 5
    livenessProbe:
      httpGet:
        path: /health
        port: 8080
      initialDelaySeconds: 15
      periodSeconds: 10
    volumeMounts:
    - name: config
      mountPath: /etc/config
    - name: tmp
      mountPath: /tmp
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
  volumes:
  - name: config
    configMap:
      name: app-config
  - name: tmp
    emptyDir: {}
  affinity:
    podAntiAffinity:
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchLabels:
              app: my-app
          topologyKey: kubernetes.io/hostname
  topologySpreadConstraints:
  - maxSkew: 1
    topologyKey: kubernetes.io/hostname
    whenUnsatisfiable: DoNotSchedule
    labelSelector:
      matchLabels:
        app: my-app
```

## Next Steps

✅ YAML syntax rules and pitfalls  
✅ Kubernetes API groups and resource structure  
✅ kubectl output formats and JSONPath  
✅ Labels, annotations, selectors  
✅ Patching and versioning  

**You have now completed all foundational concepts!**

**Continue with the main topics:**
- [01-pod-to-pod-communication.md](../01-pod-to-pod-communication.md)
- [02-service-types-loadbalancing.md](../02-service-types-loadbalancing.md)

**Recommended learning path:**
1. Pod Lifecycle → Pod Scheduling
2. Services → Ingress → Network Policies
3. ConfigMaps/Secrets → Storage
4. Autoscaling → StatefulSets
5. RBAC → Admission Controllers
6. CRDs → Custom Controllers
