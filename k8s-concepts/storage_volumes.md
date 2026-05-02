# Storage & Volumes in Kubernetes

## Overview

Kubernetes storage provides **persistent data** beyond a pod's lifetime. The storage system has three layers:

1. **PersistentVolume (PV)** - Actual storage resource (provisioned by admin or dynamically)
2. **PersistentVolumeClaim (PVC)** - Request for storage by a user/application
3. **StorageClass** - Defines how storage is dynamically provisioned

## Storage Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Application Layer                                               │
│   Pod → Mounts Volume → Reads/Writes files                      │
└───────────────────┬─────────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────────┐
│ Kubernetes API Layer                                            │
│   PersistentVolumeClaim (PVC) - "I need 10Gi of fast storage"  │
└───────────────────┬─────────────────────────────────────────────┘
                    │ Binding
┌───────────────────▼─────────────────────────────────────────────┐
│ Storage Binding Layer                                           │
│   PersistentVolume (PV) - "Here's 10Gi of SSD storage"         │
└───────────────────┬─────────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────────┐
│ Storage Backend                                                 │
│   AWS EBS / GCP PD / Azure Disk / NFS / Ceph / Local Disk      │
└─────────────────────────────────────────────────────────────────┘
```

## Volume Types Overview

| Type | Persistence | Scope | Use Case |
|------|-------------|-------|----------|
| `emptyDir` | Pod lifetime | Pod | Scratch space, inter-container |
| `hostPath` | Node lifetime | Node | Node-specific data |
| `configMap` | Until deleted | Cluster | Configuration injection |
| `secret` | Until deleted | Cluster | Sensitive config |
| `persistentVolumeClaim` | Persistent | Cluster | Databases, app state |
| `nfs` | Persistent | External | Shared access |
| `awsElasticBlockStore` | Persistent | AWS | **Deprecated** — use `ebs.csi.aws.com` CSI |
| `gcePersistentDisk` | Persistent | GCP | **Deprecated** — use `pd.csi.storage.gke.io` CSI |

> The in-tree plugins `awsElasticBlockStore`, `gcePersistentDisk`, `azureDisk`, and similar were removed or locked in Kubernetes 1.29+. All new storage integrations use CSI drivers. Use the CSI-based StorageClass (shown in the Dynamic Provisioning section below) for all production workloads.

## Static Provisioning Flow

**Admin pre-creates PVs manually.**

```
Admin creates PersistentVolume
     │
     ├─→ 1. PV created with spec
     │       accessModes: [ReadWriteOnce]
     │       capacity: 10Gi
     │       hostPath: /data/myapp
     │
     ├─→ 2. PV status: Available
     │
Developer creates PersistentVolumeClaim
     │
     ├─→ 3. PVC requests storage
     │       accessModes: [ReadWriteOnce]
     │       resources: 10Gi
     │
     ├─→ 4. Control loop: PV controller watches PVCs
     │
     ├─→ 5. Binding: PVC matched to compatible PV
     │       - Access modes compatible
     │       - Capacity sufficient
     │       - StorageClass matches (or none)
     │
     ├─→ 6. PV status: Bound
     │       PVC status: Bound
     │
     ├─→ 7. Pod references PVC in spec
     │
     └─→ 8. Kubelet mounts volume to pod
```

```yaml
# 1. Admin creates PV
apiVersion: v1
kind: PersistentVolume
metadata:
  name: my-pv
spec:
  capacity:
    storage: 10Gi
  accessModes:
  - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain  # or Delete/Recycle
  hostPath:
    path: /data/myapp
---
# 2. Developer creates PVC
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: my-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
# 3. Pod uses PVC
apiVersion: v1
kind: Pod
metadata:
  name: my-app
spec:
  containers:
  - name: app
    image: myapp:latest
    volumeMounts:
    - name: data
      mountPath: /app/data
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: my-pvc
```

## Dynamic Provisioning Flow

**StorageClass automatically creates PVs when requested.**

```
Developer creates PVC with storageClassName
     │
     ├─→ 1. PVC submitted to API server
     │
     ├─→ 2. PV controller detects PVC without bound PV
     │
     ├─→ 3. Look up StorageClass from PVC
     │       storageClassName: "fast-ssd"
     │
     ├─→ 4. Find provisioner for StorageClass
     │       provisioner: ebs.csi.aws.com
     │
     ├─→ 5. CSI provisioner pod receives request
     │       CreateVolume RPC call
     │
     ├─→ 6. Provisioner calls cloud API
     │       AWS: ec2:CreateVolume
     │       GCP: compute.disks.insert
     │
     ├─→ 7. Storage created in cloud
     │       EBS Volume ID: vol-0abc123
     │
     ├─→ 8. Provisioner creates PV object
     │       PV references EBS volume ID
     │
     ├─→ 9. PV bound to PVC
     │
     └─→ 10. Pod can now mount the volume
```

### StorageClass Examples

```yaml
# AWS EBS (SSD)
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
provisioner: ebs.csi.aws.com
parameters:
  type: gp3          # gp2, gp3, io1, io2, sc1, st1
  iopsPerGB: "50"    # For io1/io2
  encrypted: "true"
volumeBindingMode: WaitForFirstConsumer  # Multi-zone aware
reclaimPolicy: Delete  # or Retain
allowVolumeExpansion: true
---
# GCP Persistent Disk (SSD)
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd-gcp
provisioner: pd.csi.storage.gke.io
parameters:
  type: pd-ssd
  replication-type: none
reclaimPolicy: Delete
volumeBindingMode: WaitForFirstConsumer
---
# Azure Disk
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd-azure
provisioner: disk.csi.azure.com
parameters:
  skuName: Premium_LRS
reclaimPolicy: Delete
volumeBindingMode: WaitForFirstConsumer
---
# NFS (self-managed)
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: nfs-storage
provisioner: nfs.csi.k8s.io
parameters:
  server: nfs-server.default.svc.cluster.local
  share: /shared
reclaimPolicy: Retain
mountOptions:
- nfsvers=4.1
```

## Access Modes

| Mode | Abbreviation | Meaning | Use Case |
|------|-------------|---------|----------|
| ReadWriteOnce | RWO | One node can R/W | Databases |
| ReadOnlyMany | ROX | Many nodes can R | Shared config |
| ReadWriteMany | RWX | Many nodes can R/W | Shared data |
| ReadWriteOncePod | RWOP | One **pod** can R/W | High-security DBs |

```
RWO: ✅ Node1 (R/W)  ❌ Node2  ❌ Node3
ROX: ✅ Node1 (R)    ✅ Node2 (R)  ✅ Node3 (R)
RWX: ✅ Node1 (R/W)  ✅ Node2 (R/W)  ✅ Node3 (R/W)
```

## CSI (Container Storage Interface)

The modern standard for storage plugins. Replaces older in-tree plugins.

### CSI Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Kubernetes Core                                                 │
│   kubelet → csi-driver → volume operations                     │
│   kube-controller-manager → csi-provisioner → PV lifecycle    │
└──────────────────────────────┬──────────────────────────────────┘
                               │ gRPC calls
┌──────────────────────────────▼──────────────────────────────────┐
│ CSI Driver (3 components)                                       │
│                                                                  │
│  1. CSI Driver Pod (DaemonSet - 1 per node)                    │
│     ┌──────────────────────────┐                                │
│     │ node-driver-registrar    │ Register driver with kubelet   │
│     │ csi-node-driver          │ Node-specific ops (mount/umnt) │
│     │ liveness-probe           │ Health monitoring              │
│     └──────────────────────────┘                                │
│                                                                  │
│  2. CSI Controller Pod (Deployment)                             │
│     ┌──────────────────────────┐                                │
│     │ csi-provisioner          │ CreateVolume/DeleteVolume      │
│     │ csi-attacher             │ AttachVolume/DetachVolume      │
│     │ csi-resizer              │ ExpandVolume                   │
│     │ csi-snapshotter          │ CreateSnapshot                 │
│     └──────────────────────────┘                                │
│                                                                  │
│  3. CSI Driver Implementation                                   │
│     - Actual storage backend communication                      │
│     - Implements gRPC interface                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Volume Lifecycle with CSI

```
Volume Creation
     │
     ├─→ 1. PVC created with StorageClass
     │
     ├─→ 2. external-provisioner sidecar
     │       calls CSI CreateVolume
     │
     ├─→ 3. CSI driver creates EBS/PD/etc
     │
     ├─→ 4. PV created and bound to PVC
     │
Pod Scheduling
     │
     ├─→ 5. Scheduler places pod on Node1
     │
     ├─→ 6. external-attacher sidecar
     │       calls CSI ControllerPublishVolume
     │       (Attaches EBS to Node1 in AWS)
     │
     ├─→ 7. kubelet calls CSI NodeStageVolume
     │       (Format and mount to staging path)
     │
     ├─→ 8. kubelet calls CSI NodePublishVolume
     │       (Bind mount to pod directory)
     │
     └─→ 9. Pod starts with volume mounted

Pod Termination
     │
     ├─→ 10. Pod deleted
     │
     ├─→ 11. NodeUnpublishVolume (unmount from pod)
     │
     ├─→ 12. NodeUnstageVolume (unmount from node)
     │
     ├─→ 13. ControllerUnpublishVolume (detach from node)
     │
     └─→ 14. If reclaimPolicy: Delete → DeleteVolume
```

## Volume Snapshots

```yaml
# VolumeSnapshotClass
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshotClass
metadata:
  name: csi-ebs-vsc
driver: ebs.csi.aws.com
deletionPolicy: Delete
---
# Create snapshot
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: db-snapshot-2026-05-01
spec:
  volumeSnapshotClassName: csi-ebs-vsc
  source:
    persistentVolumeClaimName: database-pvc
---
# Restore from snapshot
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: restored-db
spec:
  storageClassName: fast-ssd
  dataSource:
    name: db-snapshot-2026-05-01
    kind: VolumeSnapshot
    apiGroup: snapshot.storage.k8s.io
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
```

## Storage for Databases (StatefulSets)

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  serviceName: postgres  # Headless service
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
        image: postgres:15
        env:
        - name: PGDATA
          value: /var/lib/postgresql/data/pgdata
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:  # Each pod gets its own PVC!
  - metadata:
      name: postgres-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 100Gi
```

**Result:** Creates PVCs automatically:
- `postgres-data-postgres-0`
- `postgres-data-postgres-1`
- `postgres-data-postgres-2`

Each pod gets its own dedicated volume!

## Reclaim Policies

| Policy | After PVC Delete | Use Case |
|--------|-----------------|----------|
| `Retain` | PV stays, data preserved | Critical data, manual recovery |
| `Delete` | PV and storage deleted | Ephemeral workloads |
| `Recycle` | *(deprecated)* Wiped and reused | Legacy |

```yaml
# Change reclaim policy
kubectl patch pv my-pv -p '{"spec":{"persistentVolumeReclaimPolicy":"Retain"}}'
```

## Volume Expansion

```yaml
# 1. StorageClass must allow expansion
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: ebs.csi.aws.com
allowVolumeExpansion: true  # Required!
---
# 2. Expand PVC by editing spec.resources.requests.storage
# kubectl patch pvc my-pvc -p '{"spec":{"resources":{"requests":{"storage":"50Gi"}}}}'
# or edit directly
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: my-pvc
spec:
  resources:
    requests:
      storage: 50Gi  # Increase from 10Gi
```

**Expansion Flow:**
1. PVC spec updated → storage request increased
2. CSI resizer calls `ControllerExpandVolume`
3. Cloud storage resized (EBS, PD, etc.)
4. File system resize on pod restart (or online if supported)

## Common Patterns

### Database Backup to Object Storage
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: db-backup
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:15
            command: ["/bin/sh", "-c"]
            args:
            - pg_dump -h postgres-0.postgres -U admin mydb | gzip | aws s3 cp - s3://backups/db-$(date +%Y%m%d).sql.gz
            volumeMounts:
            - name: postgres-data
              mountPath: /var/lib/postgresql/data
              readOnly: true
          volumes:
          - name: postgres-data
            persistentVolumeClaim:
              claimName: postgres-data-postgres-0
              readOnly: true
          restartPolicy: OnFailure
```

## Troubleshooting

```bash
# PVC stuck in Pending
kubectl describe pvc my-pvc
# Look for: "waiting for a volume to be created" or provisioner errors

# PV not binding
kubectl get pv
kubectl describe pv my-pv
# Check: accessModes, capacity, storageClassName must match

# Pod stuck on ContainerCreating
kubectl describe pod my-pod
# Look for: volume mount errors, PVC not bound

# Check CSI driver
kubectl get csidrivers
kubectl get csinodes

# View storage events
kubectl get events --field-selector reason=ProvisioningSucceeded
kubectl get events --field-selector reason=ProvisioningFailed
```

---

**Next:** [ConfigMaps & Secrets](07-configmaps-secrets.md)
