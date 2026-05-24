# Kubernetes SRE Interview Questions — Advanced/Expert Level

## Table of Contents
1. [Storage & Persistence](#storage--persistence)
2. [Networking](#networking)
3. [Security](#security)
4. [Performance & Optimization](#performance--optimization)
5. [High Availability & Disaster Recovery](#high-availability--disaster-recovery)
6. [Monitoring & Observability](#monitoring--observability)
7. [Resource Management](#resource-management)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Patterns](#advanced-patterns)
10. [Production Operations](#production-operations)
11. [Cost Optimization](#cost-optimization)
12. [Multi-cluster & Federation](#multi-cluster--federation)
13. [CI/CD Integration](#cicd-integration)
14. [Kubernetes Internals](#kubernetes-internals)

---

## Storage & Persistence

### Q1: Explain the difference between static and dynamic provisioning. When would you use each?

**Expected Answer:**
- **Static:** Cluster admin manually creates PVs; `provisioner: kubernetes.io/no-provisioner`
  - Use for: On-prem, custom storage, when you control hardware
  - Workflow: PV → PVC → Pod
  - No automatic PV creation; `volumeBindingMode` ignored
  
- **Dynamic:** Cloud provisioners (EBS, GCE, etc.) auto-create PVs
  - Use for: Cloud-native deployments, dev/test, flexible scaling
  - Workflow: PVC → Kubernetes creates PV → Pod
  - `volumeBindingMode` controls timing (Immediate vs WaitForFirstConsumer)

**Follow-up:** How would you implement dynamic provisioning in a local on-prem environment?
- Use LocalPV provisioner
- NFS external provisioner (nfs-subdir-external-provisioner)
- Ceph or GlusterFS with CSI drivers
- OpenEBS for container-native storage

---

### Q2: Design a multi-tier storage architecture for a microservices platform

**Expected Answer:**
- **Tier 1 - Fast (SSD):** Databases, caches (Redis), OLTP workloads
  - StorageClass: `fast` with IOPS optimization
  - Example: AWS gp3, GCP pd-ssd
  
- **Tier 2 - Standard:** Application data, logs, general workloads
  - StorageClass: `standard` with balanced performance
  - Example: AWS gp2, GCP pd-standard
  
- **Tier 3 - Archive:** Long-term backups, logs, compliance data
  - StorageClass: `archive` with S3/GCS integration
  - Retention: 7+ years
  
- **Tier 4 - Temporary:** Ephemeral data, scratch space
  - emptyDir volumes (node-local, fast)
  - No persistence required

**Implementation:**
```yaml
# Fast tier
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
reclaimPolicy: Delete
volumeBindingMode: WaitForFirstConsumer

# Archive tier with lifecycle
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: archive
provisioner: s3.csi.aws.com
parameters:
  bucket: archive-bucket
  lifecycle-rule: "transition-to-glacier:90days"
reclaimPolicy: Retain
```

**Follow-up:** How would you migrate data between tiers?
- Velero with different storage backends
- Restic for incremental backups
- Rook + Ceph for transparent data management
- Custom CronJob with data movement logic

---

### Q3: What is StatefulSet? How does it differ from Deployment when using storage?

**Expected Answer:**

| Aspect | StatefulSet | Deployment |
|--------|---|---|
| **Pod Identity** | Stable (mysql-0, mysql-1) | Random (pod-xyz) |
| **Hostname** | Stable DNS (mysql-0.mysql.svc) | Dynamic DNS |
| **Storage** | PVC per Pod via `volumeClaimTemplates` | Shared or emptyDir |
| **Update** | OrderedReady (one at a time) | RollingUpdate (parallel) |
| **Use Case** | Databases, queues, cache clusters | Stateless services |
| **Headless Service** | Required | Optional |

**When to use StatefulSet with Storage:**
- Multi-master database (PostgreSQL + streaming replication)
- RabbitMQ cluster (persistent queues per node)
- Etcd cluster (peer discovery + persistent state)
- Cassandra (consistent hashing + per-node storage)

**Example:**
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  serviceName: postgres    # ← Headless Service required
  replicas: 3
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
        - name: postgres
          image: postgres:14
          volumeMounts:
            - name: pgdata
              mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:    # ← Unique PVC per Pod
    - metadata:
        name: pgdata
      spec:
        accessModes: [ReadWriteOnce]
        storageClassName: fast
        resources:
          requests:
            storage: 50Gi
```

**Follow-up:** How would you handle data consistency when scaling a StatefulSet?
- Use ordered startup: `podManagementPolicy: Ordered`
- Implement init containers for data synchronization
- Use hooks: `lifecycle.postStart` for joining cluster
- Monitor pod readiness before next one starts

---

### Q4: Explain PVC-to-PV binding algorithm and `volumeBindingMode: WaitForFirstConsumer`

**Expected Answer:**

**Standard Binding (Immediate):**
1. PVC created → Kubernetes immediately searches for matching PV
2. Matching criteria: size, accessMode, storageClassName
3. First match wins (if multiple PVs available)
4. Problem: May bind to PV on different zone/node (affinity violation)

**WaitForFirstConsumer Binding:**
1. PVC created → Stays Pending (unbound)
2. Pod scheduled → kubelet selects node
3. Only then does binding happen → Finds PV on that node's zone
4. Benefits: Respects topology constraints (zone affinity)

**Comparison:**
```
Immediate: PVC → Bind to PV → Pod scheduled (may cross zone boundaries)
WaitForFirstConsumer: PVC → Pod scheduled → Bind to PV (respects zone)
```

**When to use WaitForFirstConsumer:**
- Multi-zone clusters (AWS, GCP, Azure)
- LocalPV or zone-pinned storage
- Cost optimization (avoid cross-zone traffic)

**Real-world scenario:**
```yaml
# ✓ Correct: Pod and PV in same zone
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: ebs-gp3
provisioner: ebs.csi.aws.com
volumeBindingMode: WaitForFirstConsumer  # Wait for Pod to schedule
allowedTopologies:                        # Restrict to zones
  - matchLabelExpressions:
      - key: topology.ebs.csi.aws.com/zone
        values: [us-east-1a, us-east-1b, us-east-1c]

# ✗ Wrong: Immediate binding may cross zones (expensive network)
volumeBindingMode: Immediate
```

**Follow-up:** What happens if no local PV exists in the Pod's zone?
- PVC remains Pending
- Pod can't be scheduled (volume pending, not ready)
- Solution: Over-provision PVs across all zones or use dynamic provisioning

---

### Q5: Design a backup/restore strategy for stateful applications

**Expected Answer:**

**Multi-layered Backup Strategy:**

1. **Continuous Replication (Hot Backup)**
   - Method: Database streaming replication
   - Example: PostgreSQL WAL archiving, MySQL binlog replication
   - RTO: Seconds, RPO: Near-zero
   - Cost: High (secondary replicas)

2. **Snapshots (Warm Backup)**
   - Method: Volume snapshots (AWS EBS, GCP Persistent Disk)
   - Frequency: Hourly/Daily
   - RTO: 5-15 minutes, RPO: 1 hour
   - Cost: Low (snapshot storage)
   ```yaml
   apiVersion: snapshot.storage.k8s.io/v1
   kind: VolumeSnapshot
   metadata:
     name: db-snapshot
   spec:
     volumeSnapshotClassName: csi-snapshotter
     source:
       persistentVolumeClaimName: db-pvc
   ```

3. **Full Backups (Cold Backup)**
   - Method: Velero (full cluster backup)
   - Frequency: Daily/Weekly
   - RTO: 30+ minutes, RPO: 1 day
   - Cost: Very low (S3/GCS storage)
   ```bash
   velero backup create prod-daily \
     --include-namespaces prod \
     --ttl 720h \
     --storage-location aws-s3
   ```

4. **Application-level Backups**
   - Method: Database dumps (pg_dump, mysqldump)
   - Frequency: Daily + transaction logs
   - RTO: 10+ minutes, RPO: Near-zero
   - Cost: Moderate

**Restore Testing (Critical):**
```bash
# Test restore in DR environment weekly
velero restore create --from-backup prod-daily --wait
# Verify data integrity
# Verify application functionality
# Measure actual RTO/RPO
```

**Implementation Matrix:**
```
RPO     | Replication | Snapshots | Full Backup | App Backup
---------|-------------|-----------|-------------|------------
seconds  | ✓          |           |             |
minutes  | ✓          | ✓        |             | ✓
hours    |            | ✓        | ✓          | ✓
days     |            |           | ✓          |
```

**Follow-up:** How would you handle cross-region DR with minimal data loss?
- Async replication to secondary region (RPO: minutes)
- Cross-region snapshots (AWS snapshot copy)
- Velero with S3 cross-region replication
- Custom logic: Application writes to primary + DR simultaneously

---

### Q6: Troubleshoot: "PVC stuck in Pending, PV available but won't bind"

**Expected Answer:**

**Diagnosis Steps:**
```bash
# Step 1: Check PVC status
kubectl describe pvc my-pvc
# Look for: Conditions, Events, Pending reason

# Step 2: Check PV status
kubectl describe pv my-pv
# Look for: Status (Available? Bound? Released?)

# Step 3: Check mismatch
kubectl get pv,pvc -o wide
# Compare: capacity, accessModes, storageClassName
```

**Common Mismatches:**

| Issue | Cause | Fix |
|-------|-------|-----|
| Size mismatch | PVC: 5Gi, PV: 1Gi | PV size ≥ PVC size |
| AccessMode mismatch | PVC: ReadOnlyMany, PV: ReadWriteOnce | PV must support PVC modes |
| StorageClass mismatch | PVC: `fast`, PV: `standard` | Must be identical |
| volumeBindingMode | WaitForFirstConsumer but no Pod | Pod must be scheduled first |
| Reclaim policy | PV: Released (after deletion) | PV stuck, needs manual cleanup |

**Real Example:**
```yaml
# ✗ PVC won't bind
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-1
spec:
  capacity: {storage: 1Gi}      # ← Too small!
  accessModes: [ReadWriteOnce]  # ← Restricted!
  storageClassName: manual

apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-1
spec:
  accessModes: [ReadWriteOnce, ReadOnlyMany]  # ← Needs both!
  storageClassName: manual
  resources:
    requests: {storage: 5Gi}  # ← PV can't satisfy

# ✓ Fix
spec:
  capacity: {storage: 10Gi}        # Match or exceed
  accessModes: [ReadWriteOnce, ReadOnlyMany]  # Include all needed
```

**Advanced Debugging:**
```bash
# Check kubelet logs on node
journalctl -u kubelet | grep -i pvc

# Check API server events
kubectl get events --sort-by='.lastTimestamp' | grep PVC

# Force debug: Create pod without PVC
kubectl run test --image=busybox -- sleep 3600
# If Pod starts, then issue is storage-specific
```

---

## Networking

### Q7: Explain Kubernetes networking layers. How do CNI plugins fit in?

**Expected Answer:**

**3 Networking Layers:**

1. **Container-to-Container (Pod):**
   - All containers in same Pod share network namespace
   - Communicate via `localhost:port`
   - Same IP address, different ports
   - Implementation: `pause` container in Pod

2. **Pod-to-Pod (Cluster):**
   - CNI (Container Network Interface) plugin manages
   - Each Pod gets unique cluster IP
   - Pods across nodes communicate directly (no NAT)
   - Example plugins: Flannel, Calico, Weave, Cilium

3. **Pod-to-External (Ingress/Egress):**
   - Services (ClusterIP, NodePort, LoadBalancer) act as proxies
   - iptables/IPVS rules on each node
   - Example: LoadBalancer exposes external IP

**CNI Plugin Responsibilities:**
```
When Pod scheduled:
1. CNI receives call: ADD <container_id> <namespace>
2. Allocates IP from IPAM (IP Address Management)
3. Creates veth pair (virtual Ethernet)
4. Connects to overlay network or bridge
5. Updates node's routing table

When Pod deleted:
1. CNI receives call: DEL <container_id>
2. Releases IP back to IPAM
3. Removes veth pair
4. Updates routing table
```

**Popular CNI Plugins:**

| Plugin | Type | Best For | Latency |
|--------|------|----------|---------|
| Flannel | Overlay | Simple, VXLAN tunneling | Medium |
| Calico | Network Policy | Security, BGP, pure networking | Low |
| Weave | Overlay | Fast setup, encryption | Medium |
| Cilium | eBPF-based | Advanced, L7 filtering, performance | Low |
| Antrea (VMware) | Overlay | Hybrid cloud | Medium |

**Network Policy Example (with Calico):**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all
spec:
  podSelector: {}  # Apply to all Pods
  policyTypes:
    - Ingress
    - Egress
  # All traffic denied by default
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-frontend
spec:
  podSelector:
    matchLabels:
      tier: backend
  policyTypes:
    - Ingress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              tier: frontend
      ports:
        - protocol: TCP
          port: 8080
```

**Follow-up:** How would you troubleshoot Pod-to-Pod connectivity?
```bash
# 1. Check CNI plugin status
kubectl get pods -n kube-system -l k8s-app=flannel

# 2. Check Pod IPs assigned
kubectl get pods -o wide

# 3. Test connectivity from one Pod
kubectl exec -it <pod1> -- ping <pod2-ip>

# 4. Check iptables/netfilter rules
iptables -L -n -t filter | grep <pod-ip>

# 5. Check CNI plugin logs
kubectl logs -n kube-system -l k8s-app=flannel

# 6. Verify routing
ip route show | grep <pod-subnet>
```

---

### Q8: Design a multi-zone Ingress architecture with auto-failover

**Expected Answer:**

**Architecture:**
```
                    Global Load Balancer (DNS)
                              |
                ______________|______________
               /              |              \
            Zone A          Zone B          Zone C
         (Primary)       (Secondary)      (Tertiary)
              |              |              |
           NLB 1           NLB 2          NLB 3
              |              |              |
        Ingress Pool 1  Ingress Pool 2  Ingress Pool 3
              |              |              |
        Nodes (Zone A)  Nodes (Zone B)  Nodes (Zone C)
              |              |              |
          Backend          Backend        Backend
          Services         Services       Services
```

**Implementation:**

1. **Global Load Balancer (DNS):**
   ```yaml
   # Using ExternalDNS + Route53
   apiVersion: v1
   kind: Service
   metadata:
     name: global-api
     annotations:
       external-dns.alpha.kubernetes.io/hostname: api.example.com
       external-dns.alpha.kubernetes.io/alias: "true"
       external-dns.alpha.kubernetes.io/aws-weight: "100"
       external-dns.alpha.kubernetes.io/aws-region: us-east-1
   spec:
     type: LoadBalancer
     selector:
       app: api
   ```

2. **Regional Ingress Controllers:**
   ```bash
   # Zone A: Primary (weight 100)
   helm install ingress-a ingress-nginx/ingress-nginx \
     --set controller.service.type=LoadBalancer \
     --set controller.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].key=topology.kubernetes.io/zone \
     --set controller.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].values[0]=us-east-1a

   # Zone B: Secondary (weight 50)
   helm install ingress-b ingress-nginx/ingress-nginx \
     --set controller.service.type=LoadBalancer \
     --set controller.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].key=topology.kubernetes.io/zone \
     --set controller.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].values[0]=us-east-1b
   ```

3. **Health-aware Routing:**
   ```yaml
   apiVersion: projectcontour.io/v1
   kind: HTTPProxy
   metadata:
     name: api-proxy
   spec:
     virtualhost:
       fqdn: api.example.com
       tls:
         secretName: api-tls
     routes:
       - conditions:
           - prefix: /
         healthStatusPolicy:
           - statusCodes: [200, 101]
             description: Healthy
           - statusCodes: [503]
             description: Degraded
         services:
           - name: api-service
             port: 8080
             healthPort: 8081  # Health check port
   ```

4. **Failover Policy:**
   ```bash
   # Route53 health check
   aws route53 create-health-check \
     --health-check-config \
     IPAddress=<nlb-ip-zone-a>,Port=80,Type=HTTP,ResourcePath=/health

   # If NLB fails, Route53 removes zone A, traffic goes to B
   ```

**Monitoring & Alerting:**
```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: ingress-failover-alerts
spec:
  groups:
    - name: ingress
      rules:
        - alert: IngressHealthCheckFailing
          expr: probe_success{ingress_zone="us-east-1a"} == 0
          for: 2m
          annotations:
            summary: "Primary Ingress unhealthy, failover triggered"
            action: "Check Zone A, may auto-failover to Zone B"
```

**RTO/RPO:**
- RTO: 30-60 seconds (Route53 health check interval)
- RPO: 0 (stateless services)

---

### Q9: Explain Service types and when to use each

**Expected Answer:**

| Type | Use Case | External Access | IP Type |
|------|----------|---|---|
| **ClusterIP** | Internal Pod-Pod comms | No (internal only) | Virtual |
| **NodePort** | Dev/test, internal services | Via Node IP:NodePort | Stable |
| **LoadBalancer** | Production external services | Via cloud LB | External |
| **ExternalName** | External service discovery | Maps to external DNS | N/A |

**Detailed Examples:**

1. **ClusterIP (Default):**
   ```yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: backend-api
   spec:
     type: ClusterIP
     selector:
       app: backend
     ports:
       - port: 80
         targetPort: 8080
   # Accessible: backend-api.default.svc.cluster.local:80
   # From same namespace: backend-api:80
   ```

2. **NodePort:**
   ```yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: frontend
   spec:
     type: NodePort
     selector:
       app: frontend
     ports:
       - port: 80           # Cluster port
         targetPort: 3000   # Pod port
         nodePort: 30000    # Node port (30000-32767)
   # Accessible: <any-node-ip>:30000
   # From cluster: frontend:80
   ```

3. **LoadBalancer (Cloud):**
   ```yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: api-lb
     annotations:
       service.beta.kubernetes.io/aws-load-balancer-type: nlb  # Network LB
   spec:
     type: LoadBalancer
     selector:
       app: api
     ports:
       - port: 443
         targetPort: 8443
         protocol: TCP
     externalTrafficPolicy: Local  # ← Preserve client IP
   # Accessible: <external-ip>:443
   ```

4. **ExternalName:**
   ```yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: external-db
   spec:
     type: ExternalName
     externalName: db.example.com
   # Usage: connect to external-db:5432 from Pod
   # Behind scenes: DNS CNAME to external host
   ```

**Advanced: Multi-cluster Service (Kubernetes 1.24+):**
```yaml
apiVersion: net.gke.io/v1
kind: MultiClusterService
metadata:
  name: api-global
spec:
  template:
    spec:
      selector:
        app: api
  clusters:
    - name: us-west1-c
      weight: 100  # Primary
    - name: us-east1-c
      weight: 50   # Secondary
```

**Follow-up:** How would you optimize LoadBalancer networking?
- Use `externalTrafficPolicy: Local` to avoid cross-node hops
- Use NLB (Network Load Balancer) for high throughput
- Use Network Endpoint Groups (NEG) for direct Pod targeting
- Implement connection pooling on client side

---

## Security

### Q10: Design a zero-trust security model for a microservices cluster

**Expected Answer:**

**Zero-Trust Principles:**
- No implicit trust (even internal traffic)
- Verify every request (authentication + authorization)
- Encrypt all traffic (mTLS)
- Least privilege access
- Continuous monitoring

**Implementation Layers:**

1. **Pod Identity & mTLS:**
   ```bash
   # Istio setup for automatic mTLS
   kubectl label namespace default istio-injection=enabled

   # All traffic encrypted automatically via sidecar proxies
   ```

2. **Network Policies (Egress + Ingress):**
   ```yaml
   apiVersion: networking.k8s.io/v1
   kind: NetworkPolicy
   metadata:
     name: api-policy
   spec:
     podSelector:
       matchLabels:
         app: api
     policyTypes:
       - Ingress
       - Egress
     ingress:
       - from:
           - podSelector:
               matchLabels:
                 role: frontend
           - namespaceSelector:
               matchLabels:
                 name: production
         ports:
           - protocol: TCP
             port: 8080
     egress:
       - to:
           - podSelector:
               matchLabels:
                 app: database
         ports:
           - protocol: TCP
             port: 5432
       - to:
           - namespaceSelector: {}
         ports:
           - protocol: TCP
             port: 53  # Allow DNS
   ```

3. **RBAC with Least Privilege:**
   ```yaml
   apiVersion: rbac.authorization.k8s.io/v1
   kind: Role
   metadata:
     name: api-reader
   rules:
     - apiGroups: [""]
       resources: ["pods"]
       verbs: ["get", "list"]
     - apiGroups: [""]
       resources: ["configmaps"]
       resourceNames: ["api-config"]  # ← Only specific resources
       verbs: ["get"]
   ---
   apiVersion: rbac.authorization.k8s.io/v1
   kind: RoleBinding
   metadata:
     name: api-reader-binding
   roleRef:
     apiGroup: rbac.authorization.k8s.io
     kind: Role
     name: api-reader
   subjects:
     - kind: ServiceAccount
       name: api-sa
       namespace: production
   ```

4. **Pod Security Policies (Deprecated → Pod Security Standards):**
   ```yaml
   apiVersion: policy/v1beta1
   kind: PodSecurityPolicy
   metadata:
     name: restricted
   spec:
     privileged: false
     allowPrivilegeEscalation: false
     requiredDropCapabilities:
       - ALL
     runAsUser:
       rule: MustRunAsNonRoot  # Can't run as UID 0
     fsGroup:
       rule: MustRunAs
       ranges:
         - min: 1000
           max: 65535
     readOnlyRootFilesystem: true
     volumes:
       - configMap
       - secret
       - emptyDir
       - persistentVolumeClaim
   ```

5. **Secret Management (Sealed Secrets):**
   ```bash
   # Install Sealed Secrets controller
   kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.18.0/controller.yaml

   # Create sealed secret
   echo -n mypassword | kubectl create secret generic mysecret --dry-run=client --from-file=password=/dev/stdin -o yaml | kubeseal -f - > mysealedsecret.yaml

   # Deploy: Sealed secret automatically decrypted by controller
   kubectl apply -f mysealedsecret.yaml
   ```

6. **Audit Logging:**
   ```yaml
   # /etc/kubernetes/audit-policy.yaml
   apiVersion: audit.k8s.io/v1
   kind: Policy
   rules:
     - level: RequestResponse
       verbs: ["create", "update", "patch", "delete"]
       resources: ["pods", "secrets", "configmaps"]
     - level: Metadata
       omitStages:
         - RequestReceived
   ```

**Security Checklist:**
```yaml
☐ mTLS enabled (Istio/Linkerd)
☐ Network Policies enforced (deny-all default)
☐ RBAC least-privilege
☐ Pod Security Standards strict
☐ Secrets encrypted at rest
☐ Image scanning in registry
☐ Container root filesystem read-only
☐ Resource limits (prevent DoS)
☐ Audit logging enabled
☐ Network traffic inspection (DPI)
☐ Runtime security monitoring (Falco)
```

---

### Q11: How would you implement multi-tenancy in Kubernetes securely?

**Expected Answer:**

**Multi-tenancy Layers:**

1. **Namespace Isolation:**
   ```yaml
   # Tenant A
   apiVersion: v1
   kind: Namespace
   metadata:
     name: tenant-a
   ---
   # Tenant B
   apiVersion: v1
   kind: Namespace
   metadata:
     name: tenant-b
   ```

2. **Network Isolation (NetworkPolicy):**
   ```yaml
   apiVersion: networking.k8s.io/v1
   kind: NetworkPolicy
   metadata:
     name: tenant-isolation
     namespace: tenant-a
   spec:
     podSelector: {}
     policyTypes:
       - Ingress
       - Egress
     ingress:
       - from:
           - podSelector: {}  # Only same-namespace Pods
     egress:
       - to:
           - podSelector: {}  # Only same-namespace Pods
       - to:
           - namespaceSelector: {}
         ports:
           - protocol: TCP
             port: 53  # Allow DNS to kube-system
   ```

3. **RBAC Isolation:**
   ```yaml
   # Tenant A service account with limited permissions
   apiVersion: v1
   kind: ServiceAccount
   metadata:
     name: tenant-a
     namespace: tenant-a
   ---
   apiVersion: rbac.authorization.k8s.io/v1
   kind: Role
   metadata:
     name: tenant-a-role
     namespace: tenant-a
   rules:
     - apiGroups: [""]
       resources: ["pods", "services"]
       verbs: ["get", "list", "create"]
       resourceNames: []  # No cluster-wide access
   ---
   apiVersion: rbac.authorization.k8s.io/v1
   kind: RoleBinding
   metadata:
     name: tenant-a-binding
     namespace: tenant-a
   roleRef:
     apiGroup: rbac.authorization.k8s.io
     kind: Role
     name: tenant-a-role
   subjects:
     - kind: ServiceAccount
       name: tenant-a
       namespace: tenant-a
   ```

4. **Resource Quotas:**
   ```yaml
   apiVersion: v1
   kind: ResourceQuota
   metadata:
     name: tenant-a-quota
     namespace: tenant-a
   spec:
     hard:
       pods: "10"
       requests.cpu: "2"
       requests.memory: "4Gi"
       limits.cpu: "4"
       limits.memory: "8Gi"
       persistentvolumeclaims: "5"
       storage: "50Gi"
   ```

5. **Pod Security Policies per Tenant:**
   ```yaml
   apiVersion: policy/v1beta1
   kind: PodSecurityPolicy
   metadata:
     name: tenant-restricted
   spec:
     privileged: false
     allowPrivilegeEscalation: false
     runAsUser:
       rule: MustRunAsNonRoot
     fsGroup:
       rule: MustRunAs
   ---
   # Bind PSP to Tenant A's ServiceAccount
   apiVersion: rbac.authorization.k8s.io/v1
   kind: Role
   metadata:
     name: psp-restricted
     namespace: tenant-a
   rules:
     - apiGroups: ["policy"]
       resources: ["podsecuritypolicies"]
       verbs: ["use"]
       resourceNames: ["tenant-restricted"]
   ```

6. **Storage Isolation (StorageClass per Tenant):**
   ```yaml
   apiVersion: storage.k8s.io/v1
   kind: StorageClass
   metadata:
     name: tenant-a-storage
   provisioner: ebs.csi.aws.com
   parameters:
     encrypted: "true"
     kmsKeyId: arn:aws:kms:us-east-1:123456789:key/tenant-a-key
   ```

7. **Cluster API Access Control:**
   ```yaml
   # kube-apiserver flags
   --authorization-mode=RBAC,Node
   --enable-admission-plugins=PodSecurityPolicy,ResourceQuota,LimitRanger
   --audit-log-path=/var/log/audit/audit.log
   --audit-policy-file=/etc/kubernetes/audit-policy.yaml
   ```

**Multi-tenancy Validation:**
```bash
# Test isolation: Tenant A shouldn't see Tenant B resources
kubectl get pods -n tenant-a --as=system:serviceaccount:tenant-a:tenant-a
# Should only show tenant-a pods

# Test network: Pod in tenant-a can't reach pod in tenant-b
kubectl exec -it <pod-a> -n tenant-a -- curl <pod-b-ip>:8080
# Should timeout
```

**Follow-up:** What are the risks of namespace-only isolation?
- Shared Kubernetes control plane (API server, etcd)
- Potential for API server vulnerabilities
- Shared node kernel
- Solution: True multi-cluster for high-security tenants, or use virtual kubelet

---

## Performance & Optimization

### Q12: Analyze cluster performance bottleneck: "Pods taking 2x longer to schedule"

**Expected Answer:**

**Investigation Steps:**

1. **Gather Metrics:**
   ```bash
   # Check scheduler latency
   kubectl get events --field-selector involvedObject.kind=Pod | grep -i warning

   # Check kube-scheduler logs
   kubectl logs -n kube-system -l component=kube-scheduler | grep -i latency

   # Prometheus queries
   histogram_quantile(0.99, rate(scheduler_scheduling_duration_seconds_bucket[5m]))
   ```

2. **Identify Root Cause:**

   | Symptom | Cause | Fix |
   |---------|-------|-----|
   | `0/3 nodes available` | Insufficient resources | Scale cluster, reduce requests |
   | `Pod pending` + taints | Node taints mismatch | Add tolerations or untaint nodes |
   | CPU 100% on scheduler | Too many Pods, complex policies | Reduce predicates, use priority queues |
   | High predicate latency | Node affinity too strict | Relax constraints |
   | Preemption happening | Pod priority inversion | Check Pod priorities |

3. **Performance Profiling:**
   ```bash
   # Get scheduler metrics
   curl http://localhost:10251/metrics | grep scheduler

   # Key metrics:
   # scheduler_scheduling_duration_seconds: Time to schedule Pod
   # scheduler_preemption_attempts_total: How many preemptions
   # scheduler_pod_scheduling_attempts: Retry count

   # Example output:
   # scheduler_scheduling_duration_seconds_bucket{le="100ms"} 500
   # scheduler_scheduling_duration_seconds_bucket{le="1000ms"} 1200
   # scheduler_scheduling_duration_seconds_bucket{le="+Inf"} 1300
   # 1.3s avg = slow!
   ```

4. **Common Bottlenecks & Solutions:**

   **Issue: Predicates too strict**
   ```yaml
   # ✗ Slow: Checking all predicates for every Pod
   apiVersion: kubescheduler.config.k8s.io/v1beta1
   kind: KubeSchedulerConfiguration
   profiles:
     - name: default-profile
       plugins:
         preFilter:
           enabled:
             - name: "NodeResourcesFit"
             - name: "NodeAffinity"
             - name: "PodTopologySpread"
             - name: "TaintToleration"

   # ✓ Fast: Use scheduling profiles with early termination
   profiles:
     - name: default-profile
       schedulingBudget:
         percentageOfNodesToScore: 10  # Check only 10% of nodes
   ```

   **Issue: Many pending Pods**
   ```bash
   # ✗ Slow: Linear search through 1000 Pods
   kubectl get pods --field-selector=status.phase=Pending | wc -l
   # 1000

   # ✓ Fast: Increase scheduler parallelism
   kube-scheduler --kube-api-qps=100 --kube-api-burst=100
   ```

   **Issue: Resource fragmentation**
   ```bash
   # ✗ Before defragmentation:
   Node1: 8 CPU used (out of 16), fragmented across 50 Pods
   Node2: 2 CPU used (out of 16), fragmented across 5 Pods
   # New Pod needs 4 CPU contiguous → can't fit!

   # ✓ After compaction:
   # Migrate Pods to Node1 → Node2 has 16 CPU free
   # New Pod fits!
   ```

5. **Optimization Config:**
   ```yaml
   apiVersion: kubescheduler.config.k8s.io/v1beta1
   kind: KubeSchedulerConfiguration
   profiles:
     - schedulerName: default-scheduler
       plugins:
         preFilter:
           disabled:
             - name: "*"  # Disable all expensive checks
           enabled:
             - name: NodeResourcesFit
         filter:
           enabled:
             - name: NodeResourcesFit
             - name: NodeAffinity
         score:
           enabled:
             - name: NodeResourcesBalancedAllocation
               weight: 3
       pluginConfig:
         - name: NodeResourcesFit
           args:
             scoringStrategy:
               type: MostAllocated  # Pack Pods densely
   ```

**Verification:**
```bash
# Before optimization
time kubectl apply -f 100-pods.yaml
# Real: 2m30s

# After optimization
time kubectl apply -f 100-pods.yaml
# Real: 1m15s ← 50% improvement!

# Monitor ongoing
kubectl top nodes
kubectl get events | grep warning
```

**Follow-up:** How would you handle priority inversion?
- Use Pod Priority Classes
- Reserve resources for critical workloads
- Implement preemption policies
- Monitor priority distribution

---

### Q13: Optimize cluster resource utilization from 30% to 85%

**Expected Answer:**

**Step 1: Analyze Current State:**
```bash
# Check node utilization
kubectl top nodes
# Typical: high variance (10%, 60%, 15%)

# Check Pod resource requests
kubectl get pods -o custom-columns=NAME:.metadata.name,REQ_CPU:.spec.containers[*].resources.requests.cpu,REQ_MEM:.spec.containers[*].resources.requests.memory

# Find overprovisioned Pods (requesting but not using)
kubectl top pods --all-namespaces | awk '{print $4, $5}' | sort -rn | head -20
```

**Step 2: Implement Right-sizing:**

```yaml
# Before (overprovisioned)
resources:
  requests:
    cpu: "2"
    memory: "2Gi"
  limits:
    cpu: "4"
    memory: "4Gi"
# Actual usage: 0.1 CPU, 256Mi memory!

# After (right-sized)
resources:
  requests:
    cpu: "100m"
    memory: "256Mi"
  limits:
    cpu: "500m"
    memory: "1Gi"
```

**Step 3: Enable Pod Density:**
```yaml
# Topology spread to pack efficiently
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
spec:
  replicas: 30  # Many small Pods
  template:
    metadata:
      labels:
        app: api
    spec:
      topologySpreadConstraints:
        - maxSkew: 2
          topologyKey: kubernetes.io/hostname
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: api
      containers:
        - name: api
          resources:
            requests:
              cpu: "50m"     # Small, dense Pods
              memory: "64Mi"
```

**Step 4: Use Bin-packing Scheduler:**
```yaml
# Scheduler config for dense packing
apiVersion: kubescheduler.config.k8s.io/v1beta1
kind: KubeSchedulerConfiguration
profiles:
  - pluginConfig:
      - name: NodeResourcesBalancedAllocation
        args:
          scoringStrategy:
            type: MostAllocated  # ← Pack tightly
            resources:
              - name: cpu
                weight: 1
              - name: memory
                weight: 1
```

**Step 5: Implement Autoscaling:**
```yaml
# HPA for workload scaling
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  minReplicas: 5
  maxReplicas: 100
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 75  # ← Keep nodes 75% full
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80

# VPA for right-sizing recommendations
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: api-vpa
spec:
  targetRef:
    apiVersion: "apps/v1"
    kind: Deployment
    name: api
  updatePolicy:
    updateMode: "auto"  # Auto-update on restart
```

**Step 6: Consolidate onto Fewer Nodes:**
```bash
# Cluster Autoscaler settings
# Scale down unused nodes after 10 minutes
--scale-down-enabled=true
--scale-down-delay-after-add=10m
--scale-down-delay-after-failure=5m
--scale-down-delay-after-delete=10s
--scale-down-unneeded-time=10m
```

**Results:**
```bash
Before:
20 nodes, 30% avg utilization
Cost: $20K/month (20 × $1K/node)

After:
6 nodes, 85% avg utilization
Cost: $6K/month

Savings: $14K/month (70% reduction!)
```

**Monitoring:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboard
data:
  utilization-dashboard.json: |
    {
      "panels": [
        {
          "title": "Cluster Utilization",
          "targets": [
            {
              "expr": "sum(rate(container_cpu_usage_seconds_total[5m])) / sum(machine_cpu_cores)",
              "legendFormat": "CPU Utilization"
            },
            {
              "expr": "sum(container_memory_working_set_bytes) / sum(machine_memory_bytes)",
              "legendFormat": "Memory Utilization"
            }
          ]
        }
      ]
    }
```

---

## High Availability & Disaster Recovery

### Q14: Design a multi-region Kubernetes disaster recovery strategy with <5 min RTO

**Expected Answer:**

**Architecture:**
```
Primary Region (us-east-1)     Secondary Region (us-west-2)
┌──────────────────────┐        ┌──────────────────────┐
│  Prod Cluster        │        │  DR Cluster (hot)    │
│  - 3 master nodes    │◄──────►│  - 3 master nodes    │
│  - 20 worker nodes   │ Async  │  - 20 worker nodes   │
│  - RTO: 0 min        │ Repl   │  - RTO: 5 min        │
└──────────────────────┘        └──────────────────────┘
         │ 100% Traffic                   │
         │                                │ 0% Traffic
    Global LB (Route53)──────────────────┼────────►
                                         │
                        Failover if primary down
```

**Step 1: Database Replication (RPO: seconds)**
```bash
# PostgreSQL Streaming Replication
Primary: postgres-prod (master)
Secondary: postgres-dr (replica)

# Configuration
wal_level = replica
max_wal_senders = 3
wal_keep_segments = 64

# Secondary follows primary continuously
# If primary fails, promote secondary to master
```

**Step 2: Application Data Synchronization**
```yaml
# Velero continuous backups to S3 (cross-region replicated)
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: prod-hourly-backup
spec:
  schedule: "@hourly"
  template:
    spec:
      includedNamespaces: ["*"]
      storageLocation: s3-primary-east
      ttl: 720h  # Keep 30 days
---
# S3 bucket with cross-region replication
aws s3api put-bucket-replication \
  --bucket velero-backups-primary \
  --replication-configuration '{
    "Role": "arn:aws:iam::123456789:role/s3-replication",
    "Rules": [{
      "Status": "Enabled",
      "Priority": 1,
      "Destination": {
        "Bucket": "arn:aws:s3:::velero-backups-secondary",
        "ReplicationTime": {"Status": "Enabled", "Time": {"Minutes": 15}}
      }
    }]
  }'
```

**Step 3: Configuration Sync (GitOps)**
```yaml
# ArgoCD watches git repo, syncs to both clusters
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: app-prod
spec:
  project: default
  source:
    repoURL: https://github.com/company/deployments.git
    targetRevision: main
    path: prod/
  destination:
    server: https://prod-cluster-api:6443
    namespace: prod
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
---
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: app-dr
spec:
  project: default
  source:
    repoURL: https://github.com/company/deployments.git
    targetRevision: main
    path: prod/
  destination:
    server: https://dr-cluster-api:6443
    namespace: prod
  syncPolicy:
    automated:
      prune: false  # Manual sync on failover
      selfHeal: false
```

**Step 4: Automated Failover Detection & Execution**
```bash
#!/bin/bash
# Monitoring script on bastion host

PRIMARY_HEALTH_CHECK="curl -f https://prod-cluster/healthz"
SECONDARY_HEALTH_CHECK="curl -f https://dr-cluster/healthz"

while true; do
  if ! $PRIMARY_HEALTH_CHECK; then
    echo "[$(date)] Primary cluster down, initiating failover..."
    
    # 1. Promote DR database (30 sec)
    kubectl --context=dr-cluster exec postgres-0 -- \
      pg_ctl promote -D /var/lib/postgresql/data
    
    # 2. Update Route53 to point to DR (instant)
    aws route53 change-resource-record-sets \
      --hosted-zone-id Z1234567890 \
      --change-batch '{
        "Changes": [{
          "Action": "UPSERT",
          "ResourceRecordSet": {
            "Name": "api.example.com",
            "Type": "A",
            "SetIdentifier": "Primary",
            "Failover": "SECONDARY",
            "TTL": 60,
            "ResourceRecords": [{"Value": "<DR-ELB-IP>"}]
          }
        }]
      }'
    
    # 3. Enable ArgoCD sync on DR (2-3 min)
    kubectl --context=dr-cluster patch application app-dr \
      -p '{"spec":{"syncPolicy":{"automated":{"prune":true,"selfHeal":true}}}}'
    
    # 4. Scale up DR workloads
    kubectl --context=dr-cluster scale deployment app --replicas=20
    
    # 5. Monitoring
    echo "Failover completed in $(date) - All traffic now to DR cluster"
    
    # Wait for recovery
    sleep 300
  fi
  
  sleep 30
done
```

**Step 5: Failback Procedure**
```bash
#!/bin/bash
# Once primary is recovered

# 1. Restore primary database from DR
pg_basebackup -h dr-postgres -D /var/lib/postgresql/data

# 2. Scale primary back up
kubectl --context=prod-cluster scale deployment app --replicas=20

# 3. Update Route53 back to primary (weighted)
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890 \
  --change-batch '{
    "Changes": [{
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "api.example.com",
        "Type": "A",
        "SetIdentifier": "Primary",
        "Weight": 100,
        "TTL": 60,
        "ResourceRecords": [{"Value": "<Primary-ELB-IP>"}]
      }
    }, {
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "api.example.com",
        "Type": "A",
        "SetIdentifier": "Secondary",
        "Weight": 0,
        "TTL": 60,
        "ResourceRecords": [{"Value": "<DR-ELB-IP>"}]
      }
    }]
  }'

# 4. Re-sync AR from git
kubectl --context=prod-cluster patch application app-prod \
  -p '{"spec":{"syncPolicy":{"automated":{"prune":true}}}}'
```

**RTO Breakdown:**
```
Primary detection (health check): 30 sec
DB promotion: 30 sec
Route53 update + DNS TTL expiration: 60-120 sec
Pod cold start on DR: 60-180 sec
Total RTO: ~5 minutes ✓
```

**Monitoring & Alerts:**
```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: dr-alerts
spec:
  groups:
    - name: disaster-recovery
      rules:
        - alert: PrimaryClusterDown
          expr: up{cluster="prod"} == 0
          for: 1m
          annotations:
            summary: "Primary cluster unresponsive, initiating failover"
        
        - alert: ReplicationLag
          expr: pg_replication_lag_seconds > 10
          for: 5m
          annotations:
            summary: "DB replication lag > 10s, may indicate sync issue"
        
        - alert: BackupMissing
          expr: time() - velero_backup_last_successful_timestamp > 3600
          for: 10m
          annotations:
            summary: "No backup in last hour, DR data stale"
```

**Testing (Monthly Disaster Recovery Drill):**
```bash
# 1. Simulate primary failure (don't actually fail)
kubectl --context=prod-cluster drain <master-node>

# 2. Trigger failover script
./failover.sh

# 3. Verify traffic on DR
curl https://api.example.com/health
# Should show DR cluster responses

# 4. Measure actual RTO
# Record time at failure detection
# Record time at successful traffic shift
# Calculate: RTO = end time - start time

# 5. Failback and verify
./failback.sh

# 6. Document findings & improvements
```

**RTO/RPO Summary:**
- **RTO:** 4-5 minutes (acceptable for most applications)
- **RPO:** ~1 minute (hourly Velero backups + 15 min S3 replication)
- **Cost:** 2x infrastructure (primary + DR running 24/7)

**Follow-up:** How would you achieve RTO < 1 minute?
- Active-active multi-region (both regions serving traffic)
- Sub-second failover via Geolocation DNS
- Pre-warmed DR with live data sync
- Extra cost: 3x infrastructure

---

### Q15: Implement backup/restore with zero downtime

**Expected Answer:**

**Multi-layer Backup Strategy:**

1. **Live Replication (Continuous):**
   ```yaml
   # Kubernetes API objects + etcd backup
   apiVersion: v1
   kind: Secret
   metadata:
     name: etcd-backup-aws
   stringData:
     AWS_ACCESS_KEY_ID: "xxx"
     AWS_SECRET_ACCESS_KEY: "yyy"
   ---
   apiVersion: batch/v1
   kind: CronJob
   metadata:
     name: etcd-backup
   spec:
     schedule: "*/5 * * * *"  # Every 5 minutes
     jobTemplate:
       spec:
         template:
           spec:
             serviceAccountName: backup-sa
             containers:
               - name: backup
                 image: bitnami/etcd:latest
                 command:
                   - /bin/sh
                   - -c
                   - |
                     etcdctl snapshot save /tmp/etcd-backup-$(date +%s).db \
                       --endpoints=https://etcd-0:2379 \
                       --cacert=/etc/kubernetes/pki/etcd/ca.crt \
                       --cert=/etc/kubernetes/pki/etcd/server.crt \
                       --key=/etc/kubernetes/pki/etcd/server.key
                     
                     aws s3 cp /tmp/etcd-backup-*.db \
                       s3://backups/etcd/ \
                       --sse=AES256
                 volumeMounts:
                   - name: etcd-certs
                     mountPath: /etc/kubernetes/pki/etcd
             volumes:
               - name: etcd-certs
                 hostPath:
                   path: /etc/kubernetes/pki/etcd
             restartPolicy: OnFailure
   ```

2. **Application Data Snapshots (Scheduled):**
   ```yaml
   apiVersion: snapshot.storage.k8s.io/v1
   kind: VolumeSnapshotClass
   metadata:
     name: aws-snapshot-class
   driver: ebs.csi.aws.com
   deletionPolicy: Delete
   ---
   apiVersion: snapshot.storage.k8s.io/v1
   kind: VolumeSnapshot
   metadata:
     name: database-snapshot
   spec:
     volumeSnapshotClassName: aws-snapshot-class
     source:
       persistentVolumeClaimName: database-pvc
   ---
   # Automated snapshots via CronJob
   apiVersion: batch/v1
   kind: CronJob
   metadata:
     name: daily-snapshots
   spec:
     schedule: "0 2 * * *"  # 2 AM daily
     jobTemplate:
       spec:
         template:
           spec:
             serviceAccountName: snapshot-creator
             containers:
               - name: snapshot
                 image: bitnami/kubectl:latest
                 command:
                   - /bin/sh
                   - -c
                   - |
                     kubectl create volumesnapshot db-snapshot-$(date +%Y%m%d) \
                       --source-name=database-pvc \
                       --snapshot-class=aws-snapshot-class
             restartPolicy: OnFailure
   ```

3. **Full Cluster Backup (Velero):**
   ```yaml
   apiVersion: velero.io/v1
   kind: BackupStorageLocation
   metadata:
     name: aws-s3
   spec:
     provider: aws
     bucket: velero-backups
     config:
       region: us-east-1
   ---
   apiVersion: velero.io/v1
   kind: Schedule
   metadata:
     name: daily-backup
   spec:
     schedule: "0 1 * * *"  # 1 AM daily
     template:
       ttl: 720h
       storageLocation: aws-s3
       includedNamespaces: ["*"]
       includedResources: ["*"]
       excludedResources:
         - nodes
         - events
         - events.events.k8s.io
   ```

**Zero-Downtime Restore:**

```yaml
# Option 1: Restore to same cluster (updates existing resources)
apiVersion: velero.io/v1
kind: Restore
metadata:
  name: restore-latest
spec:
  backupName: daily-backup-20240101
  restoreStatus:
    - downloadRequest
  namespaceMapping:
    prod: prod-temp  # Restore to temp namespace first
  itemOperationTimeout: 10m
  existingResourcePolicy: update  # ← Update, don't skip

# Option 2: Restore to different cluster
apiVersion: velero.io/v1
kind: Restore
metadata:
  name: restore-to-dr
spec:
  backupName: daily-backup-20240101
  restoreStatus:
    - downloadRequest

# Option 3: Selective restore (specific resources only)
apiVersion: velero.io/v1
kind: Restore
metadata:
  name: restore-database-only
spec:
  backupName: daily-backup-20240101
  includedNamespaces: ["prod"]
  includedResources: ["statefulsets", "persistentvolumes", "persistentvolumeclaims"]
  excludedResources: ["pods", "events"]
```

**Restore Verification:**
```bash
# 1. Check restore status
kubectl describe restore restore-latest -n velero

# 2. Verify data integrity
kubectl exec -it <pod> -- \
  pg_dump -U postgres > /tmp/restored-data.sql

# 3. Compare with backup
diff /tmp/restored-data.sql /tmp/backup-data.sql

# 4. Run integration tests
kubectl apply -f integration-tests.yaml
kubectl wait --for=condition=complete job/integration-tests --timeout=300s

# 5. Monitor for errors
kubectl logs job/integration-tests
```

**Blue-Green Deployment for Zero-Downtime Updates:**
```yaml
# Blue environment (current)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-blue
spec:
  replicas: 5
  selector:
    matchLabels:
      app: app
      version: blue
  template:
    metadata:
      labels:
        app: app
        version: blue
    spec:
      containers:
        - name: app
          image: myapp:v1.0
---
# Green environment (new)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-green
spec:
  replicas: 5
  selector:
    matchLabels:
      app: app
      version: green
  template:
    metadata:
      labels:
        app: app
        version: green
    spec:
      containers:
        - name: app
          image: myapp:v2.0
---
# Service points to blue initially
apiVersion: v1
kind: Service
metadata:
  name: app
spec:
  selector:
    app: app
    version: blue  # ← Points to blue
  ports:
    - port: 80
      targetPort: 8080
---
# Canary deployment: shift gradually
apiVersion: fluxcd.io/v1beta1
kind: Kustomization
metadata:
  name: app-canary
spec:
  serviceSelector:
    matchLabels:
      app: app
  targetPort: 8080
  routes:
    - weights:
        - value: 90  # 90% to blue
        - value: 10  # 10% to green
  # After 5 min with 0 errors:
  automaticRollout:
    enabled: true
    weights:
      - value: 50
      - value: 50
```

**Restore Success Criteria:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: restore-validation
data:
  validation-script.sh: |
    #!/bin/bash
    
    # 1. Check all Pods are running
    RUNNING=$(kubectl get pods -A --field-selector=status.phase=Running | wc -l)
    TOTAL=$(kubectl get pods -A | wc -l)
    [ "$RUNNING" -eq "$TOTAL" ] || exit 1
    
    # 2. Check all Services have endpoints
    kubectl get svc -A -o jsonpath='{.items[].status.loadBalancer.ingress[0].ip}' | grep -q "." || exit 1
    
    # 3. Check database connectivity
    kubectl exec -it postgres-0 -- psql -c "SELECT 1;" || exit 1
    
    # 4. Check application health
    curl -f http://app/health || exit 1
    
    # 5. Verify data correctness
    BACKUP_COUNT=$(curl http://backup-metadata/count)
    RESTORED_COUNT=$(psql -c "SELECT COUNT(*) FROM users;")
    [ "$BACKUP_COUNT" -eq "$RESTORED_COUNT" ] || exit 1
    
    echo "Restore validation passed!"
```

---

## Monitoring & Observability

### Q16: Design comprehensive cluster monitoring with sub-second alerting

**Expected Answer:**

**Monitoring Stack:**
```
Applications
    ↓
┌─────────────────┐
│ Prometheus      │ (5s scrape interval)
│ - Metrics DB    │
│ - 4 replicas    │
└─────────────────┘
    ↓
┌─────────────────┐      ┌──────────────┐
│ Alertmanager    │◄────►│ Slack/PagerDuty
│ - Deduplication │      │ - Incidents
│ - Grouping      │      │ - Escalation
└─────────────────┘      └──────────────┘
    ↓
┌─────────────────┐
│ Grafana         │ (visualization)
└─────────────────┘
    ↓
┌─────────────────┐
│ Loki            │ (log aggregation)
└─────────────────┘
```

**Prometheus Configuration (High Availability):**
```yaml
apiVersion: monitoring.coreos.com/v1
kind: Prometheus
metadata:
  name: prometheus-ha
spec:
  replicas: 3  # High availability
  retention: 30d
  storageSpec:
    volumeClaimTemplate:
      spec:
        storageClassName: fast
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 500Gi
  serviceMonitorSelectorNilUsesHelmValues: false
  podMonitorSelectorNilUsesHelmValues: false
  ruleSelector: {}
  resources:
    requests:
      cpu: "2"
      memory: "4Gi"
    limits:
      cpu: "4"
      memory: "8Gi"
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        - labelSelector:
            matchExpressions:
              - key: app
                operator: In
                values:
                  - prometheus
          topologyKey: kubernetes.io/hostname
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: kubelet
spec:
  selector:
    matchLabels:
      k8s-app: kubelet
  endpoints:
    - port: https-metrics
      scheme: https
      tlsConfig:
        caFile: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
      relabelings:
        - sourceLabels: [__meta_kubernetes_node_name]
          action: replace
          targetLabel: node
```

**Critical Alerts:**
```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: kubernetes-critical-alerts
spec:
  groups:
    - name: availability
      interval: 5s  # Frequent evaluation
      rules:
        # ← 1 min to detect
        - alert: KubeletDown
          expr: up{job="kubelet"} == 0
          for: 1m
          annotations:
            severity: critical
            summary: "Kubelet down on {{ $labels.node }}"
        
        - alert: MasterNodeDown
          expr: up{job="kube-apiserver"} == 0
          for: 30s  # Even faster for master
          annotations:
            severity: critical
            summary: "API Server unreachable"
        
        - alert: PVCUnbound
          expr: |
            count(kube_persistentvolumeclaim_status_phase{phase="Pending"})
              by (namespace, persistentvolumeclaim) > 0
          for: 5m
          annotations:
            severity: warning
            summary: "PVC {{ $labels.persistentvolumeclaim }} pending for 5m"
    
    - name: performance
      interval: 10s
      rules:
        - alert: HighPodMemoryUsage
          expr: |
            (sum(container_memory_working_set_bytes) by (pod, namespace) 
             / sum(container_spec_memory_limit_bytes) by (pod, namespace)) > 0.9
          for: 2m
          annotations:
            severity: warning
            summary: "Pod {{ $labels.pod }} memory usage > 90%"
        
        - alert: HighNodeCPUUsage
          expr: |
            (1 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])))) > 0.85
          for: 5m
          annotations:
            severity: warning
            summary: "Node {{ $labels.instance }} CPU > 85%"
    
    - name: reliability
      interval: 30s
      rules:
        - alert: HighErrorRate
          expr: |
            sum(rate(http_requests_total{status=~"5.."}[5m]))
              / sum(rate(http_requests_total[5m])) > 0.05
          for: 2m
          annotations:
            severity: critical
            summary: "Error rate > 5% for 2 minutes"
        
        - alert: PodCrashLoop
          expr: |
            rate(kube_pod_container_status_restarts_total[1h]) > 0
          for: 5m
          annotations:
            severity: warning
            summary: "Pod {{ $labels.pod }} crashing ({{ $value }} restarts/hour)"
```

**Alertmanager Configuration (Intelligent Routing):**
```yaml
apiVersion: monitoring.coreos.com/v1
kind: Alertmanager
metadata:
  name: alertmanager-ha
spec:
  replicas: 3
  storage:
    volumeClaimTemplate:
      spec:
        storageClassName: fast
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 10Gi
  config:
    global:
      resolve_timeout: 5m
      slack_api_url: 'xxx'
    route:
      receiver: 'default'
      group_by: ['alertname', 'cluster', 'service']
      group_wait: 10s      # Wait 10s to group alerts
      group_interval: 10s  # Resend every 10s if unresolved
      repeat_interval: 4h  # Remind every 4h
      routes:
        # Critical: immediate notification
        - match:
            severity: critical
          receiver: 'pagerduty'
          group_wait: 0s
          repeat_interval: 5m
        
        # Warning: batch in Slack
        - match:
            severity: warning
          receiver: 'slack-alerts'
          group_wait: 30s
    receivers:
      - name: 'default'
        slack_configs:
          - channel: '#alerts'
            title: '{{ .GroupLabels.alertname }}'
            text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
      
      - name: 'pagerduty'
        pagerduty_configs:
          - service_key: 'xxx'
            description: '{{ .GroupLabels.alertname }}'
            details:
              firing: '{{ range .Alerts.Firing }}{{ .Labels.instance }}{{ end }}'
      
      - name: 'slack-alerts'
        slack_configs:
          - channel: '#warnings'
    inhibit_rules:
      # Don't alert on pod restart if node is down
      - source_match:
          severity: 'critical'
          alertname: 'KubeletDown'
        target_match:
          severity: 'warning'
          alertname: 'PodCrashLoop'
        equal: ['node']
```

**Distributed Tracing (Jaeger):**
```yaml
apiVersion: jaegertracing.io/v1
kind: Jaeger
metadata:
  name: jaeger-prod
spec:
  strategy: production
  collector:
    maxReplicas: 10
  query:
    options:
      query.max-trace-duration: 1h
  storage:
    type: elasticsearch
    elasticsearch:
      nodeCount: 3
      storage:
        size: 100Gi
      resources:
        requests:
          cpu: "1"
          memory: "2Gi"
  sampling:
    type: probabilistic
    param: 0.001  # Sample 0.1% of traces
  ingestion:
    jaeger:
      agent:
        options:
          processor.zipkin-compact.server-host-port: ":5775"
```

**SLO Definition & Alerting:**
```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: slo-service-api
spec:
  groups:
    - name: slo
      rules:
        # 99.9% availability SLO
        - record: slo:api_availability:monthly
          expr: |
            sum(rate(http_requests_total{service="api", status!~"5.."}[5m]))
            / sum(rate(http_requests_total{service="api"}[5m]))
        
        - alert: APIAvailabilitySLOAtRisk
          expr: |
            slo:api_availability:monthly < 0.999
          for: 5m
          annotations:
            summary: "API availability SLO at risk: {{ $value }}"
        
        # 99th percentile latency < 500ms
        - record: slo:api_latency_p99:monthly
          expr: |
            histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{service="api"}[5m]))
        
        - alert: APILatencySLOAtRisk
          expr: |
            slo:api_latency_p99:monthly > 0.5
          for: 5m
          annotations:
            summary: "API P99 latency at risk: {{ $value }}s"
```

**Observability Best Practices:**
```bash
# 1. Monitor ALL system components
kubectl get nodes -L topology.kubernetes.io/zone
# Ensure metrics from each zone

# 2. Set appropriate scrape intervals
# Critical paths: 5s
# Normal: 30s
# Low-priority: 60s

# 3. Cardinality management (prevent OOM)
# Limit label combinations
# Remove high-cardinality labels (user_id, request_id)

# 4. Alert fatigue prevention
# Test alerting rules before deploying
# Use inhibit rules to suppress noisy alerts
# Set appropriate thresholds

# 5. Regular failover testing
# Simulate Prometheus failure monthly
# Verify alerts still fire
# Measure alerting latency
```

---

## Resource Management

### Q17: Implement resource quotas and LimitRanges for multi-tenant cluster

**Expected Answer:**

*[Content continues in next section due to length...]*

---

## Troubleshooting

### Q18: Systematic approach to troubleshooting cluster issues

**Expected Answer:**

**Troubleshooting Decision Tree:**

```
Is Pod running?
├─ NO:
│  ├─ kubectl describe pod <pod>
│  │  ├─ Pending → Check events, node resources
│  │  ├─ CrashLoopBackOff → Check logs, exit code
│  │  ├─ ImagePullBackOff → Check image registry, credentials
│  │  ├─ OOMKilled → Increase memory limit
│  │  └─ Others → Check specific error message
│  └─ kubectl logs <pod>
│  └─ kubectl logs <pod> --previous
│
└─ YES:
   ├─ Can Pod communicate?
   │  ├─ Test from Pod: kubectl exec -it <pod> -- curl service:port
   │  ├─ Check DNS: kubectl exec -it <pod> -- nslookup service
   │  ├─ Check Service endpoints: kubectl get endpoints service
   │  └─ Check NetworkPolicy: kubectl get networkpolicies
   │
   └─ Is application working?
      ├─ Check logs: kubectl logs <pod>
      ├─ Check health probes: kubectl describe pod <pod> | grep -A 5 "Liveness"
      ├─ Port forwarding: kubectl port-forward <pod> 8080:8080
      └─ Test application behavior
```

**Common Issues & Fixes:**

1. **Pod Pending:**
   ```bash
   kubectl describe pod stuck-pod
   # Events section shows reason
   
   # Case 1: Insufficient resources
   kubectl top nodes
   # If all nodes >90% full, scale cluster
   
   # Case 2: Node selector mismatch
   kubectl get nodes --show-labels
   kubectl get pod stuck-pod -o yaml | grep nodeSelector
   
   # Case 3: Taints blocking scheduling
   kubectl describe node <node> | grep Taints
   kubectl get pod stuck-pod -o yaml | grep tolerations
   ```

2. **Pod CrashLoopBackOff:**
   ```bash
   # Get previous log before container restarted
   kubectl logs <pod> --previous
   
   # Check exit code
   kubectl get event pod/<pod> -o yaml | grep reason
   
   # Run container manually to test
   kubectl run -it debug --image=<same-image> --restart=Never -- bash
   ```

3. **Service Not Reachable:**
   ```bash
   # 1. Check Service exists and has ClusterIP
   kubectl get svc myservice
   
   # 2. Check Endpoints (IPs of Pods)
   kubectl get endpoints myservice
   # Should NOT be <none>
   
   # 3. If Endpoints empty, selector mismatch
   kubectl get pods --show-labels
   kubectl get svc myservice -o yaml | grep selector
   
   # 4. Test DNS
   kubectl run -it debug --image=busybox --restart=Never -- nslookup myservice
   
   # 5. Test Pod directly (bypass Service)
   kubectl get pod -o wide
   kubectl exec -it <pod> -- curl localhost:8080
   
   # 6. Check iptables rules on node
   iptables -L -n -t filter | grep service-ip
   ```

4. **Memory Leak Detection:**
   ```bash
   # Monitor memory over time
   kubectl top pods --containers
   # Record every minute for 1 hour
   
   # Or use Prometheus
   kubectl port-forward -n monitoring prometheus-0 9090:9090
   # Query: container_memory_working_set_bytes{pod="myapp"}
   
   # If memory constantly increasing:
   # 1. Check for goroutine leaks (if Go app)
   # 2. Check for unclosed connections
   # 3. Profile heap: pprof
   ```

**Interactive Troubleshooting:**
```bash
# 1. Create debug Pod in same namespace/network
kubectl run -it debugger --image=nicolaka/netshoot -- bash

# 2. Inside debugger Pod
apt update && apt install -y postgresql mysql-client redis-tools

# 3. Test connectivity
curl http://service:8080
nc -zv database-service 5432  # Port scan
dig service                     # DNS lookup

# 4. Packet capture
tcpdump -i eth0 -n "port 8080" | head -20

# 5. Check iptables rules
iptables -L -n
iptables-save | grep <service-ip>

# 6. Check routing
ip route
netstat -tulpn
```

---

[Continue with remaining sections: Advanced Patterns, Production Operations, Cost Optimization, Multi-cluster & Federation, CI/CD Integration, Kubernetes Internals...]

---

## Advanced Patterns

### Q19: Explain Kubernetes Operators. Design a custom operator for a stateful application

**Expected Answer:**

**What is a Kubernetes Operator?**
- Custom controller that manages application lifecycle
- Encodes domain knowledge (e.g., PostgreSQL clustering, Elasticsearch sharding)
- Uses Custom Resource Definitions (CRDs) + Controller pattern
- Extends Kubernetes API with custom resources

**Example: PostgreSQL Operator**
```yaml
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: postgres-prod
spec:
  instances: 3
  postgresql:
    version: 14
  storage:
    size: 100Gi
    storageClassName: fast
  monitoring:
    enabled: true
  backup:
    retentionPolicy: "7d"
    barmanObjectStore:
      wal:
        compression: gzip
      destinationPath: s3://backups/
---
# Operator automatically handles:
# - Cluster initialization
# - Replication setup
# - Failover & recovery
# - Backup scheduling
# - Scaling
# - Monitoring
```

**Operator Architecture:**
```
Custom Resource (PostgreSQL Cluster)
         ↓
CRD (ClusterRole, ClusterRoleBinding)
         ↓
Custom Controller (watches CR changes)
         ↓
Reconciliation Loop:
  1. Check current state
  2. Compare with desired state
  3. Take action (create Pods, Services, ConfigMaps)
  4. Update status
         ↓
Managed Resources (Pods, Services, PVCs)
```

**Build Custom Operator (Kubebuilder):**
```bash
# Install kubebuilder
curl -L -o kubebuilder https://go.kubebuilder.io/dl/latest/$(go env GOOS)/$(go env GOARCH)

# Create new operator
kubebuilder init --domain example.com --repo github.com/example/redis-operator

# Create API
kubebuilder create api --group cache --version v1alpha1 --kind Redis

# Implement reconciler
# controllers/redis_controller.go
```

**Key Operators:**
- PostgreSQL: cnpg-io/cloudnative-pg
- MySQL: presslabs/mysql-operator
- MongoDB: mongodb/mongodb-kubernetes-operator
- Elasticsearch: elastic/cloud-on-k8s
- Kafka: strimzi/strimzi-kafka-operator
- Redis: spotahome/redis-operator

**Follow-up:** When would you use an Operator vs Helm chart?
- **Operator:** Complex, stateful apps, self-healing, advanced operations
- **Helm:** Simple deployments, configuration management, package distribution

---

### Q20: What are eBPF-based CNI plugins? Advantages over traditional overlay networks?

**Expected Answer:**

**eBPF (Extended Berkeley Packet Filter):**
- In-kernel virtual machine runs sandboxed programs
- Hooks at kernel level (vs userspace overlay)
- Near-native performance
- Dynamic program loading (no kernel recompile)

**Comparison:**

| Aspect | Traditional Overlay (Flannel) | eBPF (Cilium) | Performance |
|--------|---|---|---|
| **Encapsulation** | VXLAN/UDP tunnel (multiple copies) | Direct kernel forwarding | eBPF: 5-10x faster |
| **Latency** | ~0.5-1ms overhead | <0.1ms overhead | eBPF wins |
| **CPU Usage** | High (userspace processing) | Low (kernel native) | eBPF: 50% less CPU |
| **Network Policy** | Separate iptables rules | Native eBPF enforcement | eBPF: Built-in |
| **L7 Filtering** | Not supported | Full L7 visibility | eBPF only |
| **Setup** | Simple | Complex | Flannel: easier |

**eBPF Architecture (Cilium):**
```
Pod1 (Container)
  ↓
Veth pair
  ↓
eBPF Program (tc ingress hook)
  ├─ Packet inspection
  ├─ Policy enforcement
  ├─ L7 protocol parsing
  └─ Direct forwarding
  ↓
Pod2 (Container)
```

**Traditional Overlay (Flannel):**
```
Pod1 (Container)
  ↓
Veth pair
  ↓
Docker bridge
  ↓
Flannel daemon (userspace)
  ├─ VXLAN encapsulation
  ├─ Userspace packet copy
  └─ Tunnel overhead
  ↓
Remote node Flannel daemon
  ↓
Docker bridge
  ↓
Pod2 (Container)
```

**Cilium eBPF Features:**
```yaml
# L7 Network Policy (HTTP-aware)
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: api-policy
spec:
  endpointSelector:
    matchLabels:
      tier: backend
  ingress:
    - fromEndpoints:
        - matchLabels:
            tier: frontend
      toPorts:
        - ports:
            - port: "8080"
          rules:
            http:
              - method: GET
                path: /api/.*

# Deep packet inspection
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: dns-policy
spec:
  endpointSelector:
    matchLabels:
      app: dns-client
  egressDeny:
    - toPorts:
        - ports:
            - port: "53"
          rules:
            dns:
              - matchPattern: "evil\.com"  # Block specific domains!

# Service load balancing (XDP offload)
apiVersion: cilium.io/v2
kind: CiliumLoadBalancerIPPool
metadata:
  name: pool-1
spec:
  cidrs:
    - cidr: 10.0.0.0/8
  serviceSelector:
    matchLabels:
      loadBalancerIP: pool-1
```

**Performance Benchmark (Cilium vs Flannel):**
```
Throughput (Gbps):
├─ Flannel VXLAN: 5 Gbps
├─ Cilium native: 25 Gbps (5x improvement!)
└─ Cilium XDP: 40+ Gbps

Latency (microseconds):
├─ Flannel: 200 µs
├─ Cilium native: 50 µs
└─ Cilium XDP: 20 µs

CPU Usage (% at 10Gbps):
├─ Flannel: 40%
└─ Cilium: 8%
```

**When to Use eBPF:**
- ✅ High-performance networking required
- ✅ Fine-grained network policies (L7 filtering)
- ✅ Complex service mesh use cases
- ✅ Cost optimization (fewer resources)
- ❌ Older kernel (<4.9)
- ❌ Very simple deployments

---

## Production Operations

### Q21: Rolling update strategy with zero downtime

**Expected Answer:**

**Strategies:**

1. **RollingUpdate (Default):**
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: app
   spec:
     strategy:
       type: RollingUpdate
       rollingUpdate:
         maxSurge: 1        # Temporary +1 Pod
         maxUnavailable: 0  # Keep all Pods running
     replicas: 5
     minReadySeconds: 30    # Pod must be ready 30s before next update
     progressDeadlineSeconds: 600
     template:
       spec:
         containers:
           - name: app
             image: myapp:v2.0
             readinessProbe:
               httpGet:
                 path: /health
                 port: 8080
               initialDelaySeconds: 5
               periodSeconds: 5
             livenessProbe:
               httpGet:
                 path: /health
                 port: 8080
               initialDelaySeconds: 15
               periodSeconds: 10
   ```
   
   **Timeline:**
   ```
   Pod1 (v1) → Pod1 (v2, starting) → Pod1 (v2, ready)
   Pod2 (v1) → [Pod1 ready, Pod2 starting] → Pod2 (v2, ready)
   Pod3 (v1) → [Pod2 ready, Pod3 starting] → Pod3 (v2, ready)
   ...
   Total time: replicas × minReadySeconds
   ```

2. **Blue-Green Deployment:**
   ```yaml
   # Blue environment (current v1)
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: app-blue
   spec:
     replicas: 5
     selector:
       matchLabels:
         app: app
         version: blue
     template:
       spec:
         containers:
           - image: myapp:v1.0
   
   ---
   # Green environment (new v2)
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: app-green
   spec:
     replicas: 5
     selector:
       matchLabels:
         app: app
         version: green
     template:
       spec:
         containers:
           - image: myapp:v2.0
   
   ---
   # Service points to blue initially
   apiVersion: v1
   kind: Service
   metadata:
     name: app
   spec:
     selector:
       app: app
       version: blue  # ← Points to blue
   
   ---
   # Switch traffic (1 command)
   # kubectl patch svc app -p '{"spec":{"selector":{"version":"green"}}}'
   ```
   
   **Advantages:**
   - Instant switchover (no gradual transition)
   - Easy rollback (switch selector back to blue)
   - Full testing on green before switch

3. **Canary Deployment (Gradual Rollout):**
   ```yaml
   apiVersion: fluxcd.io/v1beta1
   kind: Canary
   metadata:
     name: app-canary
   spec:
     targetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: app
     progressDeadlineSeconds: 300
     service:
       port: 80
     analysis:
       interval: 1m
       threshold: 5  # Max 5 failed checks before rollback
       metrics:
         - name: error-rate
           thresholdRange:
             max: 0.05  # Fail if >5% errors
           interval: 1m
         - name: latency
           thresholdRange:
             max: 500   # Fail if p99 > 500ms
           interval: 1m
     skipAnalysis: false
     maxWeight: 100   # Gradually increase to 100%
     stepWeight: 10   # Increase by 10% every step
   ```
   
   **Traffic Shift Over Time:**
   ```
   Time  Blue  Green  
   0:00   100%   0%    (start)
   1:00   90%    10%   (step 1, check metrics)
   2:00   80%    20%   (step 2, check metrics)
   3:00   70%    30%   (step 3, check metrics)
   ...
   9:00   10%    90%   (final step)
   10:00  0%    100%   (complete)
   ```

**Ensuring Zero Downtime:**

1. **Connection Draining:**
   ```yaml
   lifecycle:
     preStop:
       exec:
         command: ["/bin/sh", "-c", "sleep 15"]  # Allow connections to finish
   terminationGracePeriodSeconds: 30  # Total time to drain
   ```

2. **Readiness Probe:**
   ```yaml
   readinessProbe:
     httpGet:
       path: /health
       port: 8080
     initialDelaySeconds: 5
     periodSeconds: 5
     failureThreshold: 3
     successThreshold: 1
   ```
   - Service only sends traffic to ready Pods
   - During update, old Pods become not ready, removed from endpoints
   - New Pods become ready, added to endpoints

3. **Pod Disruption Budget:**
   ```yaml
   apiVersion: policy/v1
   kind: PodDisruptionBudget
   metadata:
     name: app-pdb
   spec:
     minAvailable: 3  # Always keep ≥3 Pods running
     selector:
       matchLabels:
         app: app
   ```
   - Prevents scheduler from evicting too many Pods
   - Kubernetes respects PDB during updates

4. **Health Checks:**
   ```yaml
   livenessProbe:
     httpGet:
       path: /alive
       port: 8080
     initialDelaySeconds: 15
     periodSeconds: 10
     timeoutSeconds: 2
     failureThreshold: 2
   
   readinessProbe:
     httpGet:
       path: /ready
       port: 8080
     initialDelaySeconds: 5
     periodSeconds: 5
     timeoutSeconds: 2
     failureThreshold: 3
   ```

**Verification:**
```bash
# Watch deployment progress
kubectl rollout status deployment/app

# Monitor Pods during update
kubectl get pods -l app=app -w

# Check endpoints (traffic targets)
kubectl get endpoints app

# Rollback if needed
kubectl rollout undo deployment/app
```

---

### Q22: Cluster security hardening checklist

**Expected Answer:**

**Complete Hardening Checklist:**

```yaml
# 1. API SERVER HARDENING
kube-apiserver:
  - --authorization-mode=RBAC,Node
  - --enable-admission-plugins=PodSecurityPolicy,ResourceQuota,LimitRanger,ValidatingAdmissionWebhook,MutatingAdmissionWebhook
  - --disable-admission-plugins=AlwaysAdmit
  - --encryption-provider-config=/etc/kubernetes/encryption.yaml
  - --insecure-port=0
  - --secure-port=6443
  - --tls-cert-file=/etc/kubernetes/pki/apiserver.crt
  - --tls-private-key-file=/etc/kubernetes/pki/apiserver.key
  - --audit-log-path=/var/log/audit/audit.log
  - --audit-log-maxage=30
  - --audit-log-maxsize=100
  - --service-account-key-file=/etc/kubernetes/pki/sa.key
  - --kubelet-certificate-authority=/etc/kubernetes/pki/ca.crt
  - --kubelet-client-certificate=/etc/kubernetes/pki/apiserver-kubelet-client.crt
  - --kubelet-client-key=/etc/kubernetes/pki/apiserver-kubelet-client.key
  - --feature-gates=RotateKubeletServerCertificate=true

# 2. ENCRYPTION AT REST
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
  - resources:
      - secrets
      - configmaps
    providers:
      - aescbc:
          keys:
            - name: key1
              secret: <base64-32-byte-key>
      - identity: {}

# 3. RBAC - DENY ALL DEFAULT
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: restricted
rules: []  # Empty = no permissions
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: restrict-all
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: restricted
subjects:
  - kind: Group
    name: system:unauthenticated

# 4. NETWORK POLICIES - DENY ALL DEFAULT
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
  # Explicitly allow required traffic in specific namespaces

# 5. POD SECURITY POLICIES
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: restricted
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  allowedCapabilities: []
  volumes:
    - configMap
    - secret
    - emptyDir
    - persistentVolumeClaim
  hostNetwork: false
  hostIPC: false
  hostPID: false
  runAsUser:
    rule: MustRunAsNonRoot
  seLinux:
    rule: MustRunAs
    seLinuxOptions:
      level: "s0:c123,c456"
  fsGroup:
    rule: MustRunAs
  readOnlyRootFilesystem: true

# 6. KUBELET HARDENING
kubelet:
  - --anonymous-auth=false
  - --authorization-mode=Webhook
  - --client-ca-file=/etc/kubernetes/pki/ca.crt
  - --read-only-port=0
  - --protect-kernel-defaults=true
  - --make-iptables-util-chains=true
  - --feature-gates=RotateKubeletServerCertificate=true

# 7. AUDIT LOGGING
auditPolicy:
  - level: RequestResponse
    omitStages:
      - RequestReceived
    verbs:
      - create
      - update
      - patch
      - delete
    resources:
      - secrets
      - configmaps
      - pods
  - level: Metadata
    omitStages:
      - RequestReceived

# 8. IMAGE SECURITY
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
spec:
  containers:
    - name: app
      image: myapp:v1.0@sha256:abc123...  # Use SHA256 for immutability!
      imagePullPolicy: Always
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        readOnlyRootFilesystem: true
        allowPrivilegeEscalation: false
        capabilities:
          drop:
            - ALL
          add:
            - NET_BIND_SERVICE
      volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: cache
          mountPath: /app/cache
  volumes:
    - name: tmp
      emptyDir: {}
    - name: cache
      emptyDir: {}
  securityContext:
    fsGroup: 2000
    supplementalGroups: [4000]

# 9. SECRET MANAGEMENT
# Use external secret store (Vault, AWS Secrets Manager)
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
spec:
  provider:
    vault:
      server: "https://vault.example.com:8200"
      path: "secret"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "my-role"

# 10. ADMISSION WEBHOOKS
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: image-signature-verification
webhooks:
  - name: image-verification.example.com
    rules:
      - operations: ["CREATE", "UPDATE"]
        apiGroups: [""]
        apiVersions: ["v1"]
        resources: ["pods"]
    clientConfig:
      service:
        name: image-verifier
        namespace: kube-system
        path: "/verify"
      caBundle: <CA-cert>
    admissionReviewVersions: ["v1"]
    sideEffects: None

# 11. NETWORK SEGMENTATION
# Separate networks for:
# - Control plane (etcd, apiserver, controller)
# - Nodes (kubelet communication)
# - Applications (Pod traffic)
# - Management (admin access)

# 12. TLS FOR EVERYTHING
# - API server to kubelet
# - API server to etcd
# - Kubelet to API server
# - Service-to-service (mTLS via Istio)

# 13. CERTIFICATE ROTATION
apiVersion: kubeadm.k8s.io/v1beta2
kind: ClusterConfiguration
apiServer:
  certSANs:
    - "kubernetes"
    - "kubernetes.default"
    - "kubernetes.default.svc"
    - "kubernetes.default.svc.cluster.local"
    - "api.example.com"

# 14. RBAC SERVICE ACCOUNTS
apiVersion: v1
kind: ServiceAccount
metadata:
  name: app
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: app-role
rules:
  - apiGroups: [""]
    resources: ["configmaps"]
    resourceNames: ["app-config"]  # ← Only specific resource!
    verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: app-binding
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: app-role
subjects:
  - kind: ServiceAccount
    name: app
```

**Verification Commands:**
```bash
# Check API server security settings
kube-apiserver --help | grep -E "auth|tls|encrypt"

# Verify RBAC denies by default
kubectl get clusterrolebindings | grep "cluster-admin"
# Should only have one for admin user

# Test network policies
kubectl exec -it <pod-a> -- curl <pod-b>
# Should timeout if no NetworkPolicy allows it

# Check image digests (no mutable tags)
kubectl get pods -o yaml | grep "image:" | grep -v "@sha256"
# Should return nothing

# Audit sensitive actions
kubectl get events -A | grep "delete.*secret"

# Check PSP enforcement
kubectl describe psp restricted
```

---

## Cost Optimization

### Q23: Reduce cloud infrastructure costs by 50% in 6 months

**Expected Answer:**

**6-Month Cost Reduction Plan:**

**Month 1-2: Analysis & Right-sizing**
```bash
# Analyze current spending
# - Compute: $150K/month (60 nodes × $2.5K/node)
# - Storage: $40K/month (500Ti × $80/Ti)
# - Network: $10K/month (data transfer)
# Total: $200K/month

# Week 1-2: Inventory audit
kubectl get nodes -L node.kubernetes.io/instance-type
# Find overprovisioned node types (e.g., mem-optimized for CPU workloads)

# Week 3-4: Resource analysis
kubectl top nodes --no-headers | awk '{cpu+=$2; mem+=$4} END {print "Cluster: " cpu " CPU, " mem " MB"}'
# Compare allocated vs actual usage

# Results:
# - Many nodes <30% utilized
# - Some large nodes for small Pods
# - Reserved capacity unused
```

**Month 2-3: Bin-packing & Consolidation**
```yaml
# Implement right-sizing
# Before: 60 nodes
# After: 30 nodes (consolidate via bin-packing)
# Savings: $75K/month (50% reduction on compute)

apiVersion: kubescheduler.config.k8s.io/v1beta1
kind: KubeSchedulerConfiguration
profiles:
  - pluginConfig:
      - name: NodeResourcesBalancedAllocation
        args:
          scoringStrategy:
            type: MostAllocated  # ← Pack tightly
```

**Month 3-4: Reserved Instances & Spot Instances**
```bash
# AWS Reserved Instances
# Pay upfront for 1-year commitment: 40% discount

# Before: 30 nodes × $2.5K/month = $75K/month
# RI (1-year): 30 nodes × $1.5K/month = $45K/month
# Savings: $30K/month

# Spot Instances for non-critical workloads
# - Batch jobs: -70% cost
# - Dev/test environments: -80% cost
# - Non-critical services: -60% cost

kubectl taint nodes spot-node-1 workload=batch:NoSchedule
kubectl label nodes spot-node-1 instance-type=spot

# Deployment tolerates spot
spec:
  template:
    spec:
      tolerations:
        - key: workload
          operator: Equal
          value: batch
          effect: NoSchedule
      nodeSelector:
        instance-type: spot
```

**Month 4-5: Storage Optimization**
```bash
# Before: $40K/month
# - 200Ti hot storage (frequently accessed): $32K/month
# - 300Ti warm storage (occasionally accessed): $8K/month

# After optimization:
# - 100Ti hot storage (after cleanup): $8K/month (50% reduction)
# - 150Ti warm storage (archive tier): $2K/month (cheaper)
# - Implement tiered storage policies
# Savings: $38K/month

# Cleanup orphaned PVCs
kubectl get pvc -A | grep "Unbound"
kubectl delete pvc <orphaned-pvc>

# Enable compression
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: compressed-storage
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  compression: "true"  # Reduce actual disk usage

# Tiered storage
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: hot-tier
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  iops: "3000"
---
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: cold-tier
provisioner: ebs.csi.aws.com
parameters:
  type: sc1  # Cold HDD, 50% cheaper
  iops: "250"
```

**Month 5-6: Monitoring & Automation**
```yaml
# Implement cost monitoring
apiVersion: v1
kind: ConfigMap
metadata:
  name: kubecost-config
data:
  costModel.yaml: |
    computeOptimization:
      enabled: true
      savings:
        - name: "reserved-instance"
          savings: 0.40  # 40% savings vs on-demand
        - name: "spot-instance"
          savings: 0.70  # 70% savings vs on-demand

# Auto-cleanup policies
apiVersion: batch/v1
kind: CronJob
metadata:
  name: cleanup-old-pvc
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: cleanup-sa
          containers:
            - name: cleanup
              image: bitnami/kubectl:latest
              command:
                - /bin/bash
                - -c
                - |
                  # Delete PVC unused for >30 days
                  kubectl get pvc -A -o json | \
                  jq -r '.items[] | select(.metadata.creationTimestamp < "'$(date -d '30 days ago' -u +%Y-%m-%dT%H:%M:%SZ)'") | .metadata.namespace + " " + .metadata.name' | \
                  xargs -I {} kubectl delete pvc {} -n {}
          restartPolicy: OnFailure
```

**Final Results (6 months):**
```
Month 1: $200K
Month 2: $190K (monitoring, small improvements)
Month 3: $160K (consolidation: -$30K)
Month 4: $130K (RI + spot: -$30K)
Month 5: $90K (storage optimization: -$40K)
Month 6: $100K (automation overhead: +$10K)

Total Savings: $100K/month (50% reduction!)
Annual Savings: $1.2M
```

**Cost Allocation & Chargeback:**

---

### Q24: Implement cost allocation and chargeback model

**Expected Answer:**

```yaml
# 1. TAG STRATEGY
# Tag all resources by:
# - Team: backend, frontend, data-science
# - Environment: dev, staging, prod
# - Application: api, auth, payment
# - Cost-center: engineering, marketing

apiVersion: v1
kind: Pod
metadata:
  labels:
    team: backend
    env: prod
    app: api
    cost-center: "engineering"
  annotations:
    cost-allocation: "backend-team"
    project: "core-api-v2"

# 2. RESOURCE QUOTAS PER TEAM
apiVersion: v1
kind: Namespace
metadata:
  name: backend
  labels:
    team: backend
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: backend-quota
  namespace: backend
spec:
  hard:
    pods: "100"
    requests.cpu: "10"
    requests.memory: "50Gi"
    limits.cpu: "20"
    limits.memory: "100Gi"
    persistentvolumeclaims: "50"
    storage: "500Gi"

# 3. COST MODEL DEFINITION
apiVersion: v1
kind: ConfigMap
metadata:
  name: cost-model
data:
  pricing.yaml: |
    compute:
      cpu-per-hour: 0.025  # $0.025 per CPU/hour
      memory-per-gb-hour: 0.005  # $0.005 per GB/hour
    storage:
      storage-per-gb-month: 0.1  # $0.10 per GB/month
    network:
      egress-per-gb: 0.12  # $0.12 per GB outbound
    reserved:
      cpu-discount: 0.4  # 40% discount on RI
      spot-discount: 0.7  # 70% discount on spot

# 4. COST CALCULATION SCRIPT
apiVersion: batch/v1
kind: CronJob
metadata:
  name: calculate-costs
spec:
  schedule: "0 1 * * *"  # Daily at 1 AM
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: cost-calculator
          containers:
            - name: calculator
              image: python:3.9
              command:
                - python
                - |
                  import os
                  from kubernetes import client, config
                  import json
                  
                  config.load_incluster_config()
                  v1 = client.CoreV1Api()
                  
                  # Get all pods with resource requests
                  pods = v1.list_pod_for_all_namespaces()
                  
                  costs_by_team = {}
                  
                  for pod in pods.items:
                      team = pod.metadata.labels.get('team', 'unallocated')
                      
                      for container in pod.spec.containers:
                          requests = container.resources.requests or {}
                          cpu = float(requests.get('cpu', '0').replace('m', '')) / 1000
                          memory = float(requests.get('memory', '0').replace('Gi', ''))
                          
                          hourly_cost = (cpu * 0.025) + (memory * 0.005)
                          
                          if team not in costs_by_team:
                              costs_by_team[team] = 0
                          costs_by_team[team] += hourly_cost
                  
                  # Save to ConfigMap
                  with open('/tmp/costs.json', 'w') as f:
                      json.dump(costs_by_team, f)
          restartPolicy: OnFailure

# 5. BILLING DASHBOARD
apiVersion: v1
kind: Service
metadata:
  name: cost-dashboard
spec:
  selector:
    app: kubecost
  ports:
    - port: 80
      targetPort: 9090
  type: LoadBalancer

# 6. CHARGEBACK LOGIC
# Monthly report structure:
cost-report:
  backend-team:
    compute: $15000  # Pods + nodes
    storage: $2000   # PVCs
    network: $500
    total: $17500
    
  frontend-team:
    compute: $8000
    storage: $1000
    network: $300
    total: $9300

# 7. SHOW-BACK TO TEAMS
apiVersion: v1
kind: ConfigMap
metadata:
  name: team-costs-backend
data:
  monthly-report.txt: |
    Backend Team Cost Allocation
    Period: May 2024
    
    Compute: $15,000 (86% of total)
      - Production: $12,000 (160 vCPU, 640GB memory)
      - Staging: $2,000 (20 vCPU, 80GB memory)
      - Development: $1,000 (10 vCPU, 40GB memory)
    
    Storage: $2,000 (11% of total)
      - Database PVCs: $1,500
      - Cache PVCs: $300
      - Logs: $200
    
    Network: $500 (3% of total)
      - Data egress: $400
      - Load balancers: $100
    
    Total: $17,500
    
    Top 3 Cost Drivers:
    1. Production API (35%): $6,125
    2. Database cluster (28%): $4,900
    3. Redis cache (15%): $2,625
    
    Optimization Opportunities:
    - Move non-critical data to cold storage: -$300/month
    - Use spot instances for batch jobs: -$200/month
    - Consolidate redundant services: -$500/month
    Potential savings: -$1,000/month (5.7%)
```

**Chargeback Models:**

1. **Show-back (No Actual Charging):**
   - Transparent cost visibility
   - Teams understand spending
   - No budget constraints
   - Low overhead

2. **Chargeback to Teams:**
   - Deduct from team budget
   - Incentivizes cost optimization
   - Requires finance integration

3. **Charge per Resource:**
   - $X per CPU/hour
   - $Y per GB storage/month
   - Similar to cloud provider pricing

---

## Multi-cluster & Federation

### Q25: Design multi-cluster load balancing with auto-failover

[Advanced multi-cluster architecture with MCS, Submariner, and global LB...]

---

### Q26: Implement GitOps across multiple clusters

[ArgoCD/Flux setup with multi-cluster sync...]

---

## CI/CD Integration

### Q27: Implement secure container scanning in CI/CD pipeline

```yaml
# Trivy image scanning in CI
apiVersion: batch/v1
kind: Job
metadata:
  name: image-scan
spec:
  template:
    spec:
      containers:
        - name: trivy
          image: aquasec/trivy:latest
          args:
            - image
            - --severity HIGH,CRITICAL
            - --exit-code 1
            - myapp:v1.0
      restartPolicy: Never
```

---

### Q28: Blue-green deployment strategy with Kubernetes

[Covered in Q21 Rolling Updates section with complete example...]

---

## Kubernetes Internals

### Q29: Explain Kubernetes API server request lifecycle

```
1. Authentication: Check client certificate/token
2. Authorization: Check RBAC rules
3. Admission: Validate/mutate request
4. Validation: Check schema
5. Storage: Write to etcd
6. Response: Return to client
```

---

### Q30: How does kubelet work? Container runtime integration?

**kubelet lifecycle:**
```
Pod spec → CNI plugin (networking) → CRI (container runtime) → Container created
```

---

## All Kubernetes Resources Quick Reference

### Workload Resources
- **Pod**: Smallest deployable unit
- **Deployment**: Stateless applications with ReplicaSet
- **StatefulSet**: Stateful applications with stable identity
- **DaemonSet**: Run on every node (logging, monitoring)
- **Job**: Run to completion (batch tasks)
- **CronJob**: Scheduled Jobs
- **ReplicaSet**: Low-level Pod replication (use Deployment instead)

### Service & Networking
- **Service (ClusterIP)**: Internal load balancing
- **Service (NodePort)**: Node-level port mapping
- **Service (LoadBalancer)**: Cloud LB integration
- **Ingress**: HTTP/HTTPS routing (external)
- **NetworkPolicy**: Firewall rules for Pods
- **EndpointSlice**: Efficient endpoint tracking

### Storage
- **PersistentVolume (PV)**: Storage resource
- **PersistentVolumeClaim (PVC)**: Storage request
- **StorageClass**: Dynamic provisioning
- **VolumeSnapshot**: Point-in-time storage snapshots

### Configuration
- **ConfigMap**: Configuration data (non-sensitive)
- **Secret**: Sensitive data (passwords, tokens)
- **Projected Volume**: Combine multiple config sources

### Cluster Management
- **Namespace**: Logical cluster partitioning
- **ResourceQuota**: Limit resources per namespace
- **LimitRange**: Per-Pod resource limits
- **Node**: Physical/virtual machine
- **Lease**: Distributed locking (leader election)

### RBAC & Security
- **ServiceAccount**: Pod identity
- **Role**: Namespace-scoped permissions
- **RoleBinding**: Attach Role to user/ServiceAccount
- **ClusterRole**: Cluster-wide permissions
- **ClusterRoleBinding**: Cluster-wide permission binding
- **PodSecurityPolicy**: Pod security baseline (deprecated)

### Policies & Scaling
- **PodDisruptionBudget (PDB)**: Minimum Pod availability
- **HorizontalPodAutoscaler (HPA)**: Auto-scale replicas
- **VerticalPodAutoscaler (VPA)**: Right-size Pods

### Extensions
- **CustomResourceDefinition (CRD)**: Define custom resources
- **Webhook**: Intercept API requests (validation/mutation)
- **Operator**: Custom controller for application lifecycle

### Deprecated
- **PodSecurityPolicy**: Use Pod Security Standards instead
- **ReplicaSet**: Use Deployment instead

---

### Q31: Compare Pod, Deployment, StatefulSet, DaemonSet, Job

**Expected Answer:**

| Resource | Replicas | Identity | Use Case |
|----------|----------|----------|----------|
| **Pod** | N/A | Random | Debug, one-off tasks |
| **Deployment** | ✓ | Ephemeral | Stateless services (web, API) |
| **StatefulSet** | ✓ | Stable (mysql-0, mysql-1) | Stateful services (DB, cache) |
| **DaemonSet** | 1 per node | Per-node | Logging, monitoring, network |
| **Job** | 1+ (parallel) | Ephemeral | Batch, one-time tasks |
| **CronJob** | Scheduled | Ephemeral | Recurring tasks |

```yaml
# Deployment: Stateless web service
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    spec:
      containers:
        - name: nginx
          image: nginx:latest

---
# StatefulSet: Stateful database
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  serviceName: postgres  # Headless Service required
  replicas: 3
  template:
    spec:
      containers:
        - name: postgres
          image: postgres:14
          volumeMounts:
            - name: data
              mountPath: /var/lib/postgresql
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: [ReadWriteOnce]
        resources:
          requests:
            storage: 100Gi

---
# DaemonSet: One per node
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: filebeat
spec:
  selector:
    matchLabels:
      app: filebeat
  template:
    spec:
      tolerations:
        - effect: NoSchedule
          operator: Exists  # Run on all nodes, even tainted
      containers:
        - name: filebeat
          image: elastic/filebeat:latest
          volumeMounts:
            - name: logs
              mountPath: /var/log
      volumes:
        - name: logs
          hostPath:
            path: /var/log

---
# Job: Batch task
apiVersion: batch/v1
kind: Job
metadata:
  name: backup
spec:
  completions: 1  # Run once
  parallelism: 1  # Single Pod
  backoffLimit: 3
  template:
    spec:
      containers:
        - name: backup
          image: backup:1.0
      restartPolicy: Never

---
# CronJob: Scheduled task
apiVersion: batch/v1
kind: CronJob
metadata:
  name: daily-backup
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: backup
              image: backup:1.0
          restartPolicy: OnFailure
```

---

### Q32: Explain Admission Controllers & Webhooks

**Expected Answer:**

```yaml
# Validating Webhook: Reject invalid requests
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: no-privilege-escalation
webhooks:
  - name: no-privilege-escalation.example.com
    rules:
      - operations: ["CREATE", "UPDATE"]
        apiGroups: [""]
        apiVersions: ["v1"]
        resources: ["pods"]
    clientConfig:
      service:
        name: webhook-service
        namespace: webhook
        path: "/validate"
      caBundle: <base64-ca-cert>
    admissionReviewVersions: ["v1"]
    sideEffects: None
    timeoutSeconds: 5

# Webhook backend validation logic
POST /validate
{
  "apiVersion": "admission.k8s.io/v1",
  "kind": "AdmissionReview",
  "request": {
    "uid": "abc-123",
    "kind": {"group":"","version":"v1","kind":"Pod"},
    "object": {
      "spec": {
        "containers": [{
          "securityContext": {
            "allowPrivilegeEscalation": true  # ← Invalid!
          }
        }]
      }
    }
  }
}

# Response: Reject
{
  "apiVersion": "admission.k8s.io/v1",
  "kind": "AdmissionReview",
  "response": {
    "uid": "abc-123",
    "allowed": false,
    "status": {
      "message": "allowPrivilegeEscalation must be false"
    }
  }
}

---
# Mutating Webhook: Modify requests
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingWebhookConfiguration
metadata:
  name: add-init-container
webhooks:
  - name: add-init-container.example.com
    rules:
      - operations: ["CREATE"]
        apiGroups: [""]
        apiVersions: ["v1"]
        resources: ["pods"]
    clientConfig:
      service:
        name: webhook-service
        namespace: webhook
        path: "/mutate"
      caBundle: <base64-ca-cert>
    admissionReviewVersions: ["v1"]
    sideEffects: None

# Webhook mutation: Add init container for setup
# Request: Pod without init container
# Response: Same Pod with init container added
{
  "apiVersion": "admission.k8s.io/v1",
  "kind": "AdmissionReview",
  "response": {
    "uid": "abc-123",
    "allowed": true,
    "patch": [
      {
        "op": "add",
        "path": "/spec/initContainers",
        "value": [{
          "name": "setup",
          "image": "setup:1.0",
          "command": ["./setup.sh"]
        }]
      }
    ],
    "patchType": "JSONPatch"
  }
}
```

---

### Q33: How do ConfigMap & Secret differ? When to use each?

**Expected Answer:**

```yaml
# ConfigMap: Configuration data (not secret)
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  app.properties: |
    log.level=INFO
    cache.ttl=3600
    database.pool.size=20
  features.json: |
    {
      "darkMode": true,
      "betaFeatures": false
    }

---
# Secret: Sensitive data (base64 encoded, at-rest encrypted)
apiVersion: v1
kind: Secret
metadata:
  name: db-credentials
type: Opaque
data:
  username: dXNlcm5hbWU=  # base64 encoded
  password: cGFzc3dvcmQxMjM=
  connection-string: cG9zdGdyZXM6Ly91c2VyOnBhc3NAZGI6NTQzMg==

---
# Usage: Environment variables
apiVersion: v1
kind: Pod
metadata:
  name: app
spec:
  containers:
    - name: app
      image: myapp:1.0
      env:
        # From ConfigMap
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: log.level
        # From Secret
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: password

---
# Usage: Volume mount
spec:
  containers:
    - name: app
      volumeMounts:
        - name: config-volume
          mountPath: /etc/config
        - name: secret-volume
          mountPath: /etc/secrets
          readOnly: true
  volumes:
    - name: config-volume
      configMap:
        name: app-config
    - name: secret-volume
      secret:
        secretName: db-credentials
        defaultMode: 0400  # Read-only by owner
```

**Key Differences:**

| Aspect | ConfigMap | Secret |
|--------|-----------|--------|
| **Encoding** | Plain text | Base64 encoded |
| **Encryption** | Optional | At-rest encryption (if enabled) |
| **Size limit** | 1MB | 1MB |
| **Use case** | Configuration, features | Passwords, tokens, certs |
| **Visibility** | kubectl get configmap | Requires RBAC permission |
| **External store** | ConfigMap provider | Sealed Secrets, Vault |

---

### Q34: Design autoscaling strategy with HPA + VPA

**Expected Answer:**

```yaml
# HPA: Scale based on metrics (CPU, memory, custom metrics)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  minReplicas: 3
  maxReplicas: 50
  metrics:
    # Scale on CPU utilization
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    # Scale on memory utilization
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
    # Scale on custom metric (requests/sec)
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "1k"  # 1000 requests/sec per Pod
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0    # Scale up immediately
      policies:
        - type: Percent
          value: 50                     # Add 50% more Pods
          periodSeconds: 15
        - type: Pods
          value: 5                      # Or add 5 Pods max
          periodSeconds: 15
      selectPolicy: Max                 # Use whichever scales up more
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 min before scaling down
      policies:
        - type: Percent
          value: 10                     # Remove 10% of Pods
          periodSeconds: 60

---
# VPA: Right-size Pods (adjust requests/limits)
apiVersion: "autoscaling.k8s.io/v1"
kind: "VerticalPodAutoscaler"
metadata:
  name: "api-vpa"
spec:
  targetRef:
    apiVersion: "apps/v1"
    kind: Deployment
    name: api
  updatePolicy:
    updateMode: "Auto"  # Auto-update on restart
  resourcePolicy:
    containerPolicies:
      - containerName: "*"
        minAllowed:
          cpu: 50m
          memory: 64Mi
        maxAllowed:
          cpu: 2
          memory: 4Gi
        controlledResources: ["cpu", "memory"]
        controlledValues: RequestsAndLimits
```

**HPA vs VPA:**
- **HPA:** Scale # of Pods (horizontal)
- **VPA:** Adjust per-Pod resources (vertical)
- **Use Together:** HPA handles traffic spikes, VPA right-sizes Pods

---

### Q35: Explain PodDisruptionBudget (PDB) use cases

**Expected Answer:**

```yaml
# PDB: Ensure minimum Pods always running
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: api-pdb
spec:
  minAvailable: 3         # Keep ≥3 Pods running
  # OR
  # maxUnavailable: 1     # Allow max 1 Pod down
  
  selector:
    matchLabels:
      app: api
  
  unhealthyPodEvictionPolicy: IfHealthyBudget  # Evict unhealthy Pods if budget OK

---
# Use case: Cluster node drain
# Without PDB:
kubectl drain node-1
# Evicts ALL Pods immediately → Downtime!

# With PDB:
kubectl drain node-1
# Respects PDB: Only evicts if minAvailable maintained
# Waits for new Pods to start on other nodes
# Graceful, zero-downtime drain

---
# Multi-tier PDB strategy
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: critical-api-pdb
spec:
  minAvailable: 5  # Critical service: keep many running
  selector:
    matchLabels:
      tier: critical

---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: batch-pdb
spec:
  maxUnavailable: 50%  # Batch jobs: allow more disruption
  selector:
    matchLabels:
      tier: batch
```

---

## Interview Tips for SREs

1. **Ask Clarifying Questions:**
   - Cluster size? Multi-zone? Budget constraints?
   - SLA requirements? RTO/RPO targets?
   - Compliance requirements? Data residency?

2. **Show System Thinking:**
   - Consider trade-offs (availability vs cost vs complexity)
   - Discuss monitoring/observability from the start
   - Plan for failure, not just happy path

3. **Provide Real-world Examples:**
   - Reference production incidents you've handled
   - Explain lessons learned
   - Discuss metrics/monitoring for each solution

4. **Focus on Reliability:**
   - How would you test this at scale?
   - What could go wrong?
   - How would you detect failures?
   - What's the runbook?

5. **Discuss Automation:**
   - Manual steps don't scale
   - Self-healing systems preferred
   - GitOps for infrastructure as code
   - Monitoring-driven alerting

6. **Cost Awareness:**
   - Cloud resources cost money
   - Optimize resource utilization
   - Consider trade-offs (HA vs cost)
   - Implement cost monitoring

---

## Resource Coverage Summary

### ✅ Workload Resources Covered
- **Pod**: Core deployable unit
- **Deployment**: Stateless workloads (Q1, Q21, Q28)
- **StatefulSet**: Stateful workloads (Q3, Q31)
- **DaemonSet**: Node-per-node workloads (Q31)
- **Job**: Batch jobs (Q12, Q31)
- **CronJob**: Scheduled jobs (Q13)

### ✅ Service & Networking Resources Covered
- **Service**: All 4 types (Q9, Q31)
- **Ingress**: Multi-zone, failover (Q8, Q9)
- **NetworkPolicy**: Zero-trust security (Q10, Q11)
- **EndpointSlice**: Efficient tracking

### ✅ Storage Resources Covered
- **PersistentVolume**: Storage provisioning (Q1-Q6)
- **PersistentVolumeClaim**: Storage requests (Q1-Q6)
- **StorageClass**: Dynamic vs static (Q1-Q6)
- **VolumeSnapshot**: Point-in-time backups (Q5)

### ✅ Configuration Resources Covered
- **ConfigMap**: Configuration data (Q33)
- **Secret**: Sensitive data (Q33)
- **Projected Volume**: Multi-source config

### ✅ Cluster Management Resources Covered
- **Namespace**: Multi-tenancy (Q11, Q24)
- **ResourceQuota**: Namespace limits (Q17)
- **LimitRange**: Per-Pod limits (Q17)
- **Node**: Cluster capacity management

### ✅ RBAC & Security Resources Covered
- **ServiceAccount**: Pod identity (Q11, Q22)
- **Role/RoleBinding**: Namespace RBAC (Q11)
- **ClusterRole/ClusterRoleBinding**: Cluster RBAC (Q11)
- **PodSecurityPolicy**: Pod security (Q10, Q22)

### ✅ Scaling & Policies Resources Covered
- **HorizontalPodAutoscaler**: Auto-scale replicas (Q13, Q16, Q34)
- **VerticalPodAutoscaler**: Right-size Pods (Q13, Q34)
- **PodDisruptionBudget**: Disruption protection (Q21, Q35)

### ✅ Extension Resources Covered
- **CustomResourceDefinition**: Custom resources (Q19)
- **Webhook**: Admission control (Q22, Q32)
- **Operator**: Application lifecycle (Q19)

### ✅ Monitoring Resources Covered
- **Prometheus**: Metrics collection (Q16)
- **Alertmanager**: Alert routing (Q16)
- **ServiceMonitor**: Prometheus targets
- **PrometheusRule**: Alert rules (Q16)

### ✅ Advanced Topics Covered
- Disaster Recovery (Q14, Q15)
- High Availability (Q14, Q21)
- Performance Optimization (Q12, Q13)
- Security Hardening (Q10, Q11, Q22)
- Cost Optimization (Q23, Q24)
- Networking Architecture (Q7, Q8, Q20)
- Kubernetes Internals (Q29, Q30)
- CI/CD Integration (Q27, Q28)

---

## Quick Interview Preparation Checklist

**Before Interview:**
- [ ] Review Q1-Q10 (Storage, Networking, Security fundamentals)
- [ ] Review Q12-Q16 (Performance, HA, Monitoring)
- [ ] Review Q21-Q25 (Production operations)
- [ ] Practice drawing architecture diagrams
- [ ] Study your current Kubernetes setup
- [ ] Prepare 2-3 war stories from your experience

**During Interview:**
- [ ] Ask clarifying questions (cluster size, SLAs, constraints)
- [ ] Draw diagrams on whiteboard/paper
- [ ] Explain trade-offs (HA vs cost, simplicity vs features)
- [ ] Discuss monitoring/observability
- [ ] Mention failure scenarios and mitigations
- [ ] Reference real incident examples

**After Interview:**
- [ ] Review answers to difficult questions
- [ ] Document new learnings
- [ ] Update your mental model

---

## Kubernetes Resource Request/Limit Best Practices

```yaml
# Set BOTH requests and limits for all containers
containers:
  - name: app
    resources:
      # Requests: Minimum guaranteed resources
      requests:
        cpu: "100m"        # 0.1 CPU
        memory: "256Mi"    # 256MB
      # Limits: Maximum allowed resources
      limits:
        cpu: "500m"        # 0.5 CPU
        memory: "512Mi"    # 512MB

# Memory:
# - 256Mi minimum (small service)
# - 1Gi maximum (large application)
# - Requests = typical usage
# - Limits = peak usage + buffer

# CPU:
# - 50m minimum (idle service)
# - 2 cores maximum (compute intensive)
# - Requests = steady-state
# - Limits = burst capacity

# Never set just limits (no requests):
# ❌ Only limits → Pod can't be scheduled!
# ✅ Both set → Pod scheduled based on requests, limited by limits
```

---

## Common Interview Questions Not Covered

If asked during interview, refer to these key concepts:

1. **etcd**: Key-value store for all Kubernetes state
   - Critical for cluster availability
   - Requires backup/restore strategy
   - Encryption at rest recommended

2. **kube-proxy**: Node component for Service networking
   - iptables or IPVS implementation
   - Rewrites packets to Pod IPs
   - CPU intensive on large clusters

3. **kubelet**: Node component managing Pods
   - Talks to CRI (container runtime)
   - Manages Pod lifecycle
   - Reports node status to API server

4. **API Server**: Central Kubernetes hub
   - ~600 request types
   - RBAC, admission, audit
   - Horizontal scaling requires load balancing

5. **Controller Manager**: Runs various controllers
   - Replication controller (StatefulSet, Deployment, etc.)
   - Node controller (health, drain)
   - Service account controller

6. **Scheduler**: Assigns Pods to nodes
   - Filter phase (feasibility)
   - Score phase (optimization)
   - Preemption (evict lower-priority Pods)

---

## Final Interview Tips

1. **Embrace Complexity:**
   - Kubernetes is complex, it's OK to not know everything
   - Say "I don't know" rather than guessing
   - Offer to research and follow up

2. **Think Like an SRE:**
   - Reliability > performance
   - Automation > manual processes
   - Observability > visibility
   - Documentation > undocumented knowledge

3. **Discuss Failure:**
   - "What could go wrong?" is a great question
   - Describe incident response
   - Discuss post-mortems and improvements
   - Show how you prevent recurrence

4. **Cost Matters:**
   - Production clusters are expensive
   - Show awareness of cloud spending
   - Explain HA vs cost trade-offs
   - Discuss optimization opportunities

5. **Security First:**
   - Security isn't an afterthought
   - Zero-trust architecture mindset
   - RBAC from day 1
   - Encrypted secrets in transit & at rest

6. **Automation Rules:**
   - Manual processes don't scale
   - GitOps for infrastructure
   - CI/CD for applications
   - Self-healing systems preferred

---
