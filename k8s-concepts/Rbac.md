# RBAC (Role-Based Access Control) in Kubernetes

## Overview

RBAC controls **who** can do **what** to **which resources** in Kubernetes. It's the primary authorization mechanism.

## Core Concepts

```
WHO (Subject)    WHAT (Verbs)     WHICH (Resources)
   │                  │                  │
   │           get, list, watch          │
   │           create, update,           │
User/Group/ ─→ delete, patch ─→ pods, deployments,
ServiceAccount     apply             secrets, etc.
```

## RBAC Resources

| Resource | Scope | Purpose |
|----------|-------|---------|
| `Role` | Namespace | Permissions within a namespace |
| `ClusterRole` | Cluster | Cluster-wide or reusable permissions |
| `RoleBinding` | Namespace | Bind Role/ClusterRole to subject (in namespace) |
| `ClusterRoleBinding` | Cluster | Bind ClusterRole to subject (cluster-wide) |

## Subjects

| Subject | Description | Example |
|---------|-------------|---------|
| `User` | Human users (no K8s User object) | `alice@example.com` |
| `Group` | Group of users | `developers`, `system:masters` |
| `ServiceAccount` | Pod identity | `default`, `my-service-account` |

## Request Flow

```
kubectl get pods -n production
     │
     ├─→ 1. Authentication
     │       Identify: alice (from kubeconfig client cert)
     │
     ├─→ 2. Authorization (RBAC check)
     │
     │   Query: Can alice GET pods in namespace production?
     │   
     │   ├─→ Check RoleBindings in "production" namespace
     │   │     RoleBinding "dev-binding":
     │   │       subjects: [{kind: User, name: alice}]
     │   │       roleRef: {kind: Role, name: developer}
     │   │     
     │   │     Role "developer":
     │   │       rules: [{verbs: [get, list], resources: [pods]}]
     │   │       
     │   │     ✅ Match found! Alice can GET pods.
     │   │
     │   └─→ Check ClusterRoleBindings (if no namespace match)
     │
     └─→ 3. Request allowed → pods returned
```

## Role Examples

### Namespace-Scoped Role

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: developer
  namespace: production
rules:
# Workload read access
- apiGroups: ["", "apps"]
  resources: ["pods", "deployments", "replicasets", "statefulsets"]
  verbs: ["get", "list", "watch"]

# Pod operations
- apiGroups: [""]
  resources: ["pods/log"]
  verbs: ["get", "list"]
- apiGroups: [""]
  resources: ["pods/exec"]
  verbs: ["create"]

# ConfigMap read
- apiGroups: [""]
  resources: ["configmaps"]
  verbs: ["get", "list"]

# Jobs
- apiGroups: ["batch"]
  resources: ["jobs", "cronjobs"]
  verbs: ["get", "list", "watch", "create"]
```

### ClusterRole (Cluster-Wide)

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: cluster-reader
rules:
# All resources, read-only
- apiGroups: ["*"]
  resources: ["*"]
  verbs: ["get", "list", "watch"]

# Non-resource URLs (like /metrics, /healthz)
- nonResourceURLs: ["/metrics", "/healthz", "/readyz"]
  verbs: ["get"]
```

### Aggregated ClusterRoles

```yaml
# Base cluster role for custom extensions
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: custom-reader
  labels:
    rbac.example.com/aggregate-to-view: "true"  # Add to "view" role
rules:
- apiGroups: ["example.com"]
  resources: ["databases"]
  verbs: ["get", "list", "watch"]
---
# Built-in "view" ClusterRole aggregates roles matching this selector
# kubectl get clusterrole view -o yaml | grep aggregationRule
```

## RoleBinding Examples

### Bind User to Role

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: alice-developer
  namespace: production
subjects:
- kind: User
  name: alice@example.com   # Username from certificate CN or OIDC
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: developer
  apiGroup: rbac.authorization.k8s.io
```

### Bind Group to Role

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: dev-team-binding
  namespace: production
subjects:
- kind: Group
  name: developers        # Group from OIDC/LDAP or cert OU
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: developer
  apiGroup: rbac.authorization.k8s.io
```

### Bind ServiceAccount to ClusterRole

```yaml
# ServiceAccount for a pod
apiVersion: v1
kind: ServiceAccount
metadata:
  name: my-operator
  namespace: operators
---
# Grant it cluster-wide permissions
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: my-operator-binding
subjects:
- kind: ServiceAccount
  name: my-operator
  namespace: operators
roleRef:
  kind: ClusterRole
  name: cluster-admin  # Very powerful - use with care!
  apiGroup: rbac.authorization.k8s.io
```

### Grant ClusterRole in Specific Namespace (RoleBinding)

```yaml
# Reuse ClusterRole but scope to a namespace
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: view-in-staging
  namespace: staging  # Scoped to staging only
subjects:
- kind: User
  name: contractor@external.com
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole   # Reference ClusterRole...
  name: view          # ...but binding scopes it to namespace
  apiGroup: rbac.authorization.k8s.io
```

## ServiceAccount for Pods

```yaml
# Create ServiceAccount
apiVersion: v1
kind: ServiceAccount
metadata:
  name: backend-sa
  namespace: production
  annotations:
    eks.amazonaws.com/role-arn: "arn:aws:iam::123456:role/backend-role"  # IRSA (EKS)
---
# Grant permissions
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: backend-role
  namespace: production
rules:
- apiGroups: [""]
  resources: ["configmaps"]
  verbs: ["get", "list"]
- apiGroups: [""]
  resources: ["secrets"]
  resourceNames: ["backend-secret"]  # ONLY this specific secret
  verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: backend-binding
  namespace: production
subjects:
- kind: ServiceAccount
  name: backend-sa
roleRef:
  kind: Role
  name: backend-role
  apiGroup: rbac.authorization.k8s.io
---
# Pod uses ServiceAccount
apiVersion: v1
kind: Pod
metadata:
  name: backend-pod
  namespace: production
spec:
  serviceAccountName: backend-sa  # Use this SA
  automountServiceAccountToken: true  # Mount token (default: true)
  containers:
  - name: backend
    image: myapp:latest
```

**Inside the pod:**
```bash
# Token mounted automatically at:
cat /var/run/secrets/kubernetes.io/serviceaccount/token

# App uses this to call K8s API
APISERVER=https://kubernetes.default.svc
TOKEN=$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)
curl -H "Authorization: Bearer $TOKEN" $APISERVER/api/v1/namespaces/production/configmaps
```

## Common RBAC Patterns

### Multi-Tier Permissions

```yaml
# Tier 1: Read-only for all developers
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: dev-readonly
  namespace: production
subjects:
- kind: Group
  name: developers
roleRef:
  kind: ClusterRole
  name: view  # Built-in: read-only to non-sensitive resources
---
# Tier 2: Deploy rights for senior devs
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: deployer
  namespace: production
rules:
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "update", "patch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: senior-dev-deployer
  namespace: production
subjects:
- kind: Group
  name: senior-developers
roleRef:
  kind: Role
  name: deployer
---
# Tier 3: Admin rights for SRE team
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: sre-admin
  namespace: production
subjects:
- kind: Group
  name: sre-team
roleRef:
  kind: ClusterRole
  name: admin  # Built-in: full namespace admin
```

### Namespace Admin Pattern

```yaml
# Give team full ownership of their namespace
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: team-namespace-admin
  namespace: team-a
subjects:
- kind: Group
  name: team-a-developers
roleRef:
  kind: ClusterRole
  name: admin  # Full namespace control
  apiGroup: rbac.authorization.k8s.io
```

## Built-in Cluster Roles

| ClusterRole | Access Level |
|-------------|-------------|
| `cluster-admin` | Everything in the cluster |
| `admin` | Full namespace admin (except ResourceQuota) |
| `edit` | Create/update/delete most resources |
| `view` | Read-only access (no Secrets) |

```bash
# View what each built-in role can do
kubectl describe clusterrole view
kubectl describe clusterrole edit
kubectl describe clusterrole admin
```

## Verbs Reference

| Verb | HTTP Method | Description |
|------|-------------|-------------|
| `get` | GET (single) | Read one resource |
| `list` | GET (collection) | List resources |
| `watch` | GET + watch | Watch for changes |
| `create` | POST | Create resource |
| `update` | PUT | Replace resource |
| `patch` | PATCH | Partial update |
| `delete` | DELETE | Delete resource |
| `deletecollection` | DELETE (collection) | Delete multiple |
| `use` | - | Use PodSecurityPolicy |
| `impersonate` | - | Impersonate user/group/SA |
| `bind` | - | Bind roles |
| `escalate` | - | Create roles with more permissions than you have |

## Wildcard Permissions

```yaml
rules:
# ALL verbs on ALL resources in ALL apiGroups
- apiGroups: ["*"]
  resources: ["*"]
  verbs: ["*"]
```

> ⚠️ This is `cluster-admin` level. Use sparingly!

## Resource Names (Specific Objects)

```yaml
# Access only specific secrets
rules:
- apiGroups: [""]
  resources: ["secrets"]
  resourceNames: ["database-password", "api-key"]  # ONLY these
  verbs: ["get"]

# Access only specific configmaps
- apiGroups: [""]
  resources: ["configmaps"]
  resourceNames: ["app-config"]
  verbs: ["get", "update"]
```

## OIDC Integration (External Auth)

```bash
# kube-apiserver OIDC configuration
--oidc-issuer-url=https://accounts.google.com
--oidc-client-id=kubernetes
--oidc-username-claim=email
--oidc-groups-claim=groups
```

**Flow:**
```
User logs in → Google OIDC → JWT token (contains email + groups)
     │
     ├─→ kubectl configured with OIDC token
     │
     ├─→ API server validates JWT with Google's public keys
     │
     ├─→ Extracts: username=alice@example.com, groups=[developers]
     │
     └─→ RBAC checks permissions for alice@example.com and developers group
```

## Auditing RBAC

```bash
# Check if current user can do something
kubectl auth can-i create pods
kubectl auth can-i create pods --namespace production
kubectl auth can-i create pods --as alice@example.com
kubectl auth can-i create pods --as system:serviceaccount:production:backend-sa

# Check all permissions for a user
kubectl auth can-i --list --as alice@example.com --namespace production

# Verify what a ServiceAccount can do
kubectl auth can-i list pods --as system:serviceaccount:production:backend-sa

# Find what cluster roles exist
kubectl get clusterroles | grep -v system

# Find all role bindings for a user
kubectl get rolebindings -A -o json | \
  jq '.items[] | select(.subjects[]?.name == "alice@example.com")'

# Find all clusterrolebindings for a group
kubectl get clusterrolebindings -o json | \
  jq '.items[] | select(.subjects[]?.name == "developers")'
```

## Common Security Mistakes

### 1. Overly Broad Wildcards
```yaml
# BAD
rules:
- apiGroups: ["*"]
  resources: ["*"]
  verbs: ["*"]

# GOOD - Minimal required permissions
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list"]
```

### 2. Forgetting Secrets Are Sensitive
```yaml
# BAD - "view" cluster role doesn't include secrets, but this does
rules:
- apiGroups: [""]
  resources: ["*"]  # Includes secrets!
  verbs: ["get", "list"]

# GOOD - Explicitly list resources
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps"]  # NOT secrets
  verbs: ["get", "list"]
```

### 3. Granting Escalation
```yaml
# DANGEROUS - Can create roles with more permissions than caller
rules:
- apiGroups: ["rbac.authorization.k8s.io"]
  resources: ["roles", "clusterroles", "rolebindings"]
  verbs: ["*"]  # Can escalate privileges!
```

### 4. Default ServiceAccount
```yaml
# BAD - Using default SA (might have unexpected permissions)
spec:
  # No serviceAccountName specified

# GOOD - Create dedicated SA with minimal permissions
spec:
  serviceAccountName: backend-sa
  automountServiceAccountToken: false  # If you don't need K8s API access
```

## Troubleshooting RBAC

```bash
# 403 Forbidden error?
kubectl auth can-i <verb> <resource> --as <user>

# Example:
kubectl auth can-i create deployments --as alice@example.com -n production
# no - missing permissions

# Find what's blocking it
kubectl describe rolebinding -n production
kubectl describe clusterrolebinding

# Get all RBAC for a namespace
kubectl get roles,rolebindings -n production

# Audit log for RBAC denials
# (requires audit logging enabled)
# Look for "Forbidden" responses in audit logs
```
