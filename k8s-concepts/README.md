# Kubernetes Concepts

A collection of explanations for core Kubernetes concepts with flows, diagrams, and real-world examples.

## 📚 Topics

### Networking
1. [Pod-to-Pod Communication](01-pod-to-pod-communication.md) - How pods communicate within and across nodes
2. [Service Types & Load Balancing](02-service-types-loadbalancing.md) - ClusterIP, NodePort, LoadBalancer flows
3. [Ingress & External Traffic](03-ingress-external-traffic.md) - How external requests reach your pods
4. [DNS Resolution](04-dns-resolution.md) - Service discovery with CoreDNS
5. [Network Policies](05-network-policies.md) - Traffic filtering and security

### Storage
6. [Storage & Volumes](06-storage-volumes.md) - PV, PVC, StorageClass provisioning

### Configuration
7. [ConfigMaps & Secrets](07-configmaps-secrets.md) - Configuration injection patterns

### Workload Management
8. [Pod Lifecycle](08-pod-lifecycle.md) - From creation to termination
9. [Pod Scheduling](09-pod-scheduling.md) - How scheduler places pods
10. [Autoscaling](10-autoscaling.md) - HPA, VPA, Cluster Autoscaler

### Advanced Topics
11. [CRDs & Custom Controllers](11-crds-controllers.md) - Extending Kubernetes
12. [API Gateway Patterns](12-api-gateway.md) - API management in K8s
13. [Admission Controllers & Webhooks](13-admission-controllers.md) - Request validation and mutation
14. [RBAC](14-rbac.md) - Role-Based Access Control
15. [StatefulSets](15-statefulsets.md) - Stateful application patterns
16. [Jobs & CronJobs](16-jobs-cronjobs.md) - Batch workload execution

## 🎯 How to Use

Each document follows a consistent structure:
- **Core Concepts** - Fundamental principles
- **Flow Diagrams** - Visual representation of processes
- **Step-by-Step Explanations** - Detailed breakdowns
- **Real-World Examples** - YAML manifests and scenarios
- **Performance Considerations** - Best practices and optimization

## 🔗 Quick Navigation

**New to Kubernetes?** Start with:
1. Pod Lifecycle → Pod-to-Pod Communication → DNS Resolution → Services

**Networking Deep Dive:**
Pod-to-Pod → Services → Ingress → Network Policies → DNS

**Storage Workflows:**
Storage & Volumes → StatefulSets

**Extending Kubernetes:**
CRDs & Controllers → Admission Controllers
