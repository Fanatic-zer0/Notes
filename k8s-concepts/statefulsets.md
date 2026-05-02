# StatefulSets in Kubernetes

## Overview

**StatefulSets** manage stateful applications where each pod has a **stable, unique identity** that persists across rescheduling. Unlike Deployments where pods are interchangeable.

## StatefulSet vs Deployment

| Feature | Deployment | StatefulSet |
|---------|------------|-------------|
| Pod names | Random (nginx-abc123) | Predictable (mysql-0, mysql-1) |
| Storage | Shared or ephemeral | Each pod gets own PVC |
| Start order | Parallel | Sequential (0, 1, 2, ...) |
| Stop order | Random | Reverse sequential (2, 1, 0) |
| Stable network ID | No | Yes (via headless service) |
| Use case | Stateless apps | Databases, queues, caches |

## Architecture

```
StatefulSet: mysql (replicas: 3)
     │
     ├─→ mysql-0 → mysql-data-mysql-0 (PVC) → EBS vol-001
     │             network: mysql-0.mysql.default.svc.cluster.local
     │
     ├─→ mysql-1 → mysql-data-mysql-1 (PVC) → EBS vol-002
     │             network: mysql-1.mysql.default.svc.cluster.local
     │
     └─→ mysql-2 → mysql-data-mysql-2 (PVC) → EBS vol-003
                   network: mysql-2.mysql.default.svc.cluster.local

Headless Service (mysql):
     - mysql-0.mysql:3306
     - mysql-1.mysql:3306
     - mysql-2.mysql:3306
```

## StatefulSet YAML

```yaml
# 1. Headless Service (required for stable DNS)
apiVersion: v1
kind: Service
metadata:
  name: mysql
  labels:
    app: mysql
spec:
  clusterIP: None  # Headless!
  ports:
  - name: mysql
    port: 3306
  selector:
    app: mysql
---
# 2. Client Service (optional, for client connections)
apiVersion: v1
kind: Service
metadata:
  name: mysql-read
spec:
  ports:
  - name: mysql
    port: 3306
  selector:
    app: mysql
---
# 3. StatefulSet
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mysql
spec:
  serviceName: mysql       # Must match headless service name
  replicas: 3
  selector:
    matchLabels:
      app: mysql
  
  template:
    metadata:
      labels:
        app: mysql
    spec:
      initContainers:
      # Clone data from primary or previous pod
      - name: init-mysql
        image: mysql:8.0
        command:
        - bash
        - "-c"
        - |
          set -ex
          # Determine pod ordinal (0, 1, 2, ...)
          [[ $(hostname) =~ -([0-9]+)$ ]] && ordinal=${BASH_REMATCH[1]}
          
          # mysql-0 is primary, others are replicas
          if [[ $ordinal -eq 0 ]]; then
            echo "Primary node - no init needed"
          else
            echo "Replica $ordinal - cloning from mysql-0"
            mysqldump -h mysql-0.mysql -u root -p$MYSQL_ROOT_PASSWORD mydb | \
              mysql -u root -p$MYSQL_ROOT_PASSWORD mydb
          fi
        env:
        - name: MYSQL_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mysql-secret
              key: root-password
      
      containers:
      - name: mysql
        image: mysql:8.0
        ports:
        - containerPort: 3306
        env:
        - name: MYSQL_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mysql-secret
              key: root-password
        
        volumeMounts:
        - name: mysql-data
          mountPath: /var/lib/mysql
        - name: config
          mountPath: /etc/mysql/conf.d
        
        readinessProbe:
          exec:
            command: ["mysqladmin", "ping"]
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
        
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: "2"
            memory: 4Gi
      
      volumes:
      - name: config
        configMap:
          name: mysql-config
  
  # Each pod gets its own PVC - auto-created
  volumeClaimTemplates:
  - metadata:
      name: mysql-data
    spec:
      accessModes: [ReadWriteOnce]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 100Gi
```

## Pod Identity and DNS

```
StatefulSet: mysql, namespace: default, headless service: mysql

Pod Names:
  mysql-0, mysql-1, mysql-2

Pod DNS:
  mysql-0.mysql.default.svc.cluster.local
  mysql-1.mysql.default.svc.cluster.local
  mysql-2.mysql.default.svc.cluster.local

Pod Hostname (inside container):
  $ hostname
  mysql-0

Pod Subdomain (for DNS):
  $ hostname -f
  mysql-0.mysql.default.svc.cluster.local
```

**This persists across restarts:**
```
mysql-0 crashes → rescheduled to different node → SAME name, SAME PVC, SAME DNS
```

## Startup and Shutdown Order

### Scale Up (Sequential)

```
kubectl scale statefulset mysql --replicas=3
     │
     ├─→ Start mysql-0
     │       Wait: mysql-0 Running & Ready
     │
     ├─→ Start mysql-1
     │       Wait: mysql-1 Running & Ready
     │
     └─→ Start mysql-2
              (mysql-0 and mysql-1 already running)
```

**Why sequential?** In primary-replica setups, primary (pod-0) must be running before replicas join.

### Scale Down (Reverse Sequential)

```
kubectl scale statefulset mysql --replicas=1
     │
     ├─→ Delete mysql-2
     │       Wait: mysql-2 fully terminated
     │
     ├─→ Delete mysql-1
     │       Wait: mysql-1 fully terminated
     │
     └─→ mysql-0 remains (last survivor = primary)
```

## Update Strategies

### RollingUpdate (Default)

```yaml
spec:
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      partition: 0  # Update all pods (default)
      # partition: 2  # Only update pods >= 2
      maxUnavailable: 1  # (Kubernetes 1.24+)
```

**Partition-based canary updates:**
```yaml
# Phase 1: Update only mysql-2 (ordinal >= 2)
updateStrategy:
  rollingUpdate:
    partition: 2

# Phase 2: If successful, update mysql-1 and mysql-2
updateStrategy:
  rollingUpdate:
    partition: 1

# Phase 3: Update all pods
updateStrategy:
  rollingUpdate:
    partition: 0
```

### OnDelete

```yaml
spec:
  updateStrategy:
    type: OnDelete  # Only update when manually deleted
```

**Usage:**
```bash
kubectl rollout status statefulset/mysql  # Doesn't work with OnDelete
kubectl delete pod mysql-2  # Manually trigger update
```

## Real-World: MySQL Primary-Replica

```yaml
# ConfigMap for MySQL configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: mysql-config
data:
  primary.cnf: |
    [mysqld]
    log-bin                      # Enable binary logging
    server-id=1                  # Unique server ID (pod-0)
    
  replica.cnf: |
    [mysqld]
    super-read-only              # Replicas are read-only
    server-id=2                  # Must be different per replica
```

**Ordinal-based config selection:**
```bash
# In init container
ordinal=$(echo $HOSTNAME | grep -o '[0-9]*$')
if [ "$ordinal" -eq "0" ]; then
  cp /config/primary.cnf /etc/mysql/conf.d/server-id.cnf
else
  # Calculate unique server-id for each replica
  echo "[mysqld]
  super-read-only
  server-id=$((100 + $ordinal))" > /etc/mysql/conf.d/server-id.cnf
fi
```

## Real-World: Redis Cluster

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
spec:
  serviceName: redis
  replicas: 6  # 3 primaries + 3 replicas
  template:
    spec:
      containers:
      - name: redis
        image: redis:7.0
        command: ["redis-server", "/conf/redis.conf"]
        ports:
        - containerPort: 6379
          name: client
        - containerPort: 16379
          name: gossip
        volumeMounts:
        - name: data
          mountPath: /data
        - name: config
          mountPath: /conf
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: [ReadWriteOnce]
      resources:
        requests:
          storage: 10Gi
```

## StatefulSet Operations

### Scaling

```bash
# Scale up (adds sequentially: mysql-3, mysql-4)
kubectl scale statefulset mysql --replicas=5

# Scale down (removes in reverse: mysql-4, mysql-3)
kubectl scale statefulset mysql --replicas=3

# Watch scaling progress
kubectl get pods -l app=mysql -w
```

### Rolling Update

```bash
# Update image
kubectl set image statefulset/mysql mysql=mysql:8.0.32

# Watch update (happens in reverse order: mysql-2, mysql-1, mysql-0)
kubectl rollout status statefulset/mysql

# Rollback
kubectl rollout undo statefulset/mysql
```

### Debugging

```bash
# Access specific pod
kubectl exec -it mysql-0 -- mysql -u root -p

# View ordinal-specific logs
kubectl logs mysql-0 --previous

# Force restart a specific pod
kubectl delete pod mysql-1  # StatefulSet recreates it

# Check PVCs
kubectl get pvc -l app=mysql
```

## PVC Lifecycle with StatefulSets

**PVCs are NOT deleted when pod is deleted or StatefulSet is scaled down!**

```bash
# Scale down (pods deleted, PVCs remain!)
kubectl scale statefulset mysql --replicas=1
kubectl get pvc  # Still shows mysql-data-mysql-1, mysql-data-mysql-2

# Scale back up (reattaches existing PVCs with their data!)
kubectl scale statefulset mysql --replicas=3
# mysql-1 and mysql-2 get their original PVCs back
```

**To delete PVCs:**
```bash
# Must manually delete PVCs after deleting StatefulSet
kubectl delete statefulset mysql
kubectl delete pvc mysql-data-mysql-0 mysql-data-mysql-1 mysql-data-mysql-2

# Or delete all PVCs with label
kubectl delete pvc -l app=mysql
```

## Headless Service vs Regular Service

**For StatefulSet, usually need BOTH:**

```yaml
# Headless Service: Stable DNS names for each pod
apiVersion: v1
kind: Service
metadata:
  name: mysql         # mysql-0.mysql, mysql-1.mysql, etc.
spec:
  clusterIP: None     # Headless
  selector:
    app: mysql

# Client Service: Load balance reads across replicas
apiVersion: v1
kind: Service
metadata:
  name: mysql-read
spec:
  selector:
    app: mysql        # All pods
    role: replica     # Or specific label if set

# Write Service: Only primary
apiVersion: v1
kind: Service
metadata:
  name: mysql-write
spec:
  selector:
    app: mysql
    role: primary
```

## Comparison: When to Use StatefulSet

### Use Deployment (stateless):
- Web servers
- API services
- Workers with external state (queue consumers)
- Any app that's truly stateless

### Use StatefulSet:
- **Databases** (PostgreSQL, MySQL, MongoDB)
- **Distributed caches** (Redis Cluster, Memcached)
- **Message brokers** (Kafka, RabbitMQ)
- **Search engines** (Elasticsearch)
- **Coordination services** (ZooKeeper, etcd)


