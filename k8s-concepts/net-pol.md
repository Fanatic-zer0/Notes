# Network Policies in Kubernetes

## Overview

**NetworkPolicy** is a Kubernetes resource that controls traffic flow at the IP address or port level (OSI Layer 3/4). Think of it as **firewall rules for your pods**.

## Default Behavior (No NetworkPolicy)

**Without NetworkPolicies:**
- ✅ All pods can communicate with each other
- ✅ All pods can reach external endpoints
- ✅ All external traffic can reach pods (if exposed)

**With NetworkPolicies:**
- ❌ Default DENY all (when a policy selects a pod)
- ✅ Only explicitly allowed traffic passes

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Kubernetes Cluster                                          │
│                                                              │
│  NetworkPolicy Resource (API Object)                        │
│       │                                                      │
│       ├─→ Stored in etcd                                    │
│       │                                                      │
│       ↓                                                      │
│  CNI Plugin (must support NetworkPolicy)                    │
│  - Calico, Cilium, Weave, Canal                            │
│  - NOT Flannel (no NetworkPolicy support)                  │
│       │                                                      │
│       ├─→ Watches NetworkPolicy resources                   │
│       ├─→ Translates to iptables/eBPF rules                │
│       │                                                      │
│       ↓                                                      │
│  Enforced at Pod Network Namespace Level                    │
│  - Ingress filtering (incoming traffic)                     │
│  - Egress filtering (outgoing traffic)                      │
└─────────────────────────────────────────────────────────────┘
```

## Basic NetworkPolicy Structure

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: example-policy
  namespace: default
spec:
  # Which pods this policy applies to
  podSelector:
    matchLabels:
      app: backend
  
  # Policy types to enforce
  policyTypes:
  - Ingress  # Control incoming traffic
  - Egress   # Control outgoing traffic
  
  # Ingress rules (who can connect TO these pods)
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8080
  
  # Egress rules (where these pods can connect TO)
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: database
    ports:
    - protocol: TCP
      port: 5432
```

## Traffic Flow with NetworkPolicy

### Without NetworkPolicy
```
┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│ Frontend Pod │ ────→  │ Backend Pod  │ ────→  │ Database Pod │
│ (any)        │        │ (any)        │        │ (any)        │
└──────────────┘        └──────────────┘        └──────────────┘
        ↕                       ↕                       ↕
   Internet                Internet               Internet
```
**Everything can talk to everything.**

### With NetworkPolicy
```
┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│ Frontend Pod │ ──✅→  │ Backend Pod  │ ──✅→  │ Database Pod │
│              │        │ [Policy]     │        │ [Policy]     │
└──────────────┘        └──────────────┘        └──────────────┘
        ↕                       ❌                      ❌
   Internet                Internet               Internet
        ↓                       ↓                       ↓
  Only outbound           Blocked               Blocked
```
**Only explicitly allowed connections work.**

## Common Patterns

### 1. Deny All Ingress Traffic

**Default DENY** - Block all incoming traffic to selected pods.

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all-ingress
  namespace: production
spec:
  podSelector: {}  # Selects ALL pods in namespace
  policyTypes:
  - Ingress
  # No ingress rules = deny all
```

### 2. Allow Only from Same Namespace

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-same-namespace
  namespace: production
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector: {}  # Any pod in same namespace
```

### 3. Frontend → Backend → Database

**Database Policy:** Only backend can connect
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: database-policy
  namespace: production
spec:
  podSelector:
    matchLabels:
      tier: database
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          tier: backend
    ports:
    - protocol: TCP
      port: 5432
```

**Backend Policy:** Only frontend can connect
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: backend-policy
  namespace: production
spec:
  podSelector:
    matchLabels:
      tier: backend
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          tier: frontend
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          tier: database
    ports:
    - protocol: TCP
      port: 5432
  - to:  # Allow DNS
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: UDP
      port: 53
```

**Frontend Policy:** Allow from LoadBalancer
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: frontend-policy
  namespace: production
spec:
  podSelector:
    matchLabels:
      tier: frontend
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - ipBlock:
        cidr: 0.0.0.0/0  # Allow from anywhere (for LB)
    ports:
    - protocol: TCP
      port: 80
  egress:
  - to:
    - podSelector:
        matchLabels:
          tier: backend
    ports:
    - protocol: TCP
      port: 8080
  - to:  # Allow DNS
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: UDP
      port: 53
```

## Selectors

### podSelector

**Same namespace pods:**
```yaml
ingress:
- from:
  - podSelector:
      matchLabels:
        app: frontend
```

### namespaceSelector

**Pods from specific namespaces:**
```yaml
ingress:
- from:
  - namespaceSelector:
      matchLabels:
        environment: production
```

### Combined Selectors

**AND logic (both must match):**
```yaml
ingress:
- from:
  - namespaceSelector:
      matchLabels:
        environment: production
    podSelector:
      matchLabels:
        app: frontend
```
**Meaning:** Pods labeled `app=frontend` from namespaces labeled `environment=production`

**OR logic (separate items):**
```yaml
ingress:
- from:
  - namespaceSelector:
      matchLabels:
        environment: production
  - podSelector:
      matchLabels:
        app: admin
```
**Meaning:** Pods from production namespaces OR pods labeled `app=admin` (any namespace)

### ipBlock

**Allow specific IP ranges:**
```yaml
ingress:
- from:
  - ipBlock:
      cidr: 192.168.1.0/24
      except:
      - 192.168.1.5/32  # Exclude specific IP
```

**Use Cases:**
- Allow traffic from specific subnets
- Allow external monitoring systems
- Whitelist office IPs

## Egress Control

### Allow DNS Only

**Critical:** Most policies need DNS for service discovery.

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-dns-only
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
  - Egress
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: UDP
      port: 53
```

### Allow External API

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-external-api
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
  - Egress
  egress:
  - to:
    - ipBlock:
        cidr: 203.0.113.0/24  # External API IP range
    ports:
    - protocol: TCP
      port: 443
  - to:  # DNS
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: UDP
      port: 53
```

## Complete Example: 3-Tier Application

```yaml
# 1. Default deny all in namespace
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: app
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
---
# 2. Frontend policy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: frontend-policy
  namespace: app
spec:
  podSelector:
    matchLabels:
      tier: frontend
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:  # Backend
    - podSelector:
        matchLabels:
          tier: backend
    ports:
    - protocol: TCP
      port: 8080
  - to:  # DNS
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: kube-system
    ports:
    - protocol: UDP
      port: 53
---
# 3. Backend policy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: backend-policy
  namespace: app
spec:
  podSelector:
    matchLabels:
      tier: backend
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          tier: frontend
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:  # Database
    - podSelector:
        matchLabels:
          tier: database
    ports:
    - protocol: TCP
      port: 5432
  - to:  # Redis cache
    - podSelector:
        matchLabels:
          tier: cache
    ports:
    - protocol: TCP
      port: 6379
  - to:  # DNS
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: kube-system
    ports:
    - protocol: UDP
      port: 53
  - to:  # External payment API
    - ipBlock:
        cidr: 198.51.100.0/24
    ports:
    - protocol: TCP
      port: 443
---
# 4. Database policy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: database-policy
  namespace: app
spec:
  podSelector:
    matchLabels:
      tier: database
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          tier: backend
    ports:
    - protocol: TCP
      port: 5432
  egress:
  - to:  # DNS only
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: kube-system
    ports:
    - protocol: UDP
      port: 53
---
# 5. Cache (Redis) policy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: cache-policy
  namespace: app
spec:
  podSelector:
    matchLabels:
      tier: cache
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          tier: backend
    ports:
    - protocol: TCP
      port: 6379
```

## Advanced Patterns

### Allow Monitoring

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-monitoring
  namespace: app
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 9090  # Prometheus scrape port
```

### Namespace Isolation

**Prevent cross-namespace communication:**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-other-namespaces
  namespace: production
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector: {}  # Only same namespace
```

### Allow Health Checks

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-health-checks
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
  - Ingress
  ingress:
  - from:
    - ipBlock:
        cidr: 10.0.0.0/8  # Node network
    ports:
    - protocol: TCP
      port: 8080
      endPort: 8090  # Health check port range
```

## CNI Plugin Implementations

### Calico
- **Method:** iptables or eBPF
- **Features:** Full NetworkPolicy support, egress/ingress
- **Performance:** Excellent

```bash
# Install Calico
kubectl apply -f https://docs.projectcalico.org/manifests/calico.yaml

# View Calico policies
calicoctl get networkpolicy
```

### Cilium
- **Method:** eBPF (in-kernel)
- **Features:** L7 policies, DNS-aware, identity-based
- **Performance:** Best in class

```yaml
# L7 HTTP policy example (Cilium-specific)
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: http-only
spec:
  endpointSelector:
    matchLabels:
      app: backend
  ingress:
  - fromEndpoints:
    - matchLabels:
        app: frontend
    toPorts:
    - ports:
      - port: "80"
        protocol: TCP
      rules:
        http:
        - method: "GET"
          path: "/api/.*"
```

### Weave
- **Method:** iptables
- **Features:** Full NetworkPolicy support
- **Performance:** Good

## Troubleshooting

### Test Connectivity

```bash
# Create test pods
kubectl run frontend --image=busybox --labels=app=frontend -- sleep 3600
kubectl run backend --image=busybox --labels=app=backend -- sleep 3600

# Test connection
kubectl exec frontend -- wget -O- http://backend:8080 --timeout=2
```

### Check Policy Application

```bash
# List policies
kubectl get networkpolicy

# Describe policy
kubectl describe networkpolicy backend-policy

# Check pod labels (policy selector)
kubectl get pods --show-labels
```

### Common Issues

**1. Policy Not Working:**
- Check CNI plugin supports NetworkPolicy (Flannel doesn't!)
- Verify pod labels match policy selector
- Ensure namespace labels are correct

**2. DNS Not Working:**
```yaml
# Add DNS egress rule
egress:
- to:
  - namespaceSelector:
      matchLabels:
        kubernetes.io/metadata.name: kube-system
  ports:
  - protocol: UDP
    port: 53
```

**3. Inter-Namespace Communication:**
```bash
# Check namespace labels
kubectl get namespace production --show-labels

# Add label if missing
kubectl label namespace production environment=production
```

### Debugging with Calico

```bash
# View effective policies
calicoctl get workloadendpoint -o yaml

# View policy order
calicoctl get globalnetworkpolicy --output=yaml
```

## Best Practices

1. **Start with default deny:**
   ```yaml
   podSelector: {}
   policyTypes: [Ingress, Egress]
   # No rules = deny all
   ```

2. **Always allow DNS:**
   ```yaml
   egress:
   - to:
     - namespaceSelector:
         matchLabels:
           kubernetes.io/metadata.name: kube-system
     ports:
     - protocol: UDP
       port: 53
   ```

3. **Use namespace isolation** for multi-tenant clusters

4. **Label everything** consistently for easy policy management

5. **Test policies** in staging before production

6. **Monitor denied connections** using CNI plugin logs

7. **Document policies** with annotations:
   ```yaml
   metadata:
     annotations:
       description: "Allows frontend to access backend on port 8080"
   ```

8. **Use least privilege** - only allow required connections

---

**Next:** [Storage & Volumes](06-storage-volumes.md)
