### Pod Lifecycle Issues
- **CrashLoopBackOff** — get logs from previous crashed container, extract core dumps, check exit codes (OOM=137, segfault=139)
- **OOMKilled** — identify memory limits, get heap dumps before container dies, use `kubectl top pod`
- **Pod stuck in `Terminating`** — force delete, remove stuck finalizers, `--grace-period=0 --force`
- **Init container failures** — exec into init containers, check init logs separately
- **`ImagePullBackOff`** — debug registry auth, ECR token expiry, image digest mismatches
- **Pod stuck in `ContainerCreating`** — CNI plugin failure, volume mount timeout, secret/configmap missing
- **`RunContainerError`** — entrypoint not found, permission denied on binary, wrong architecture (arm vs amd64)
- **Pod restarts with exit code 0** — liveness probe too aggressive, process exits cleanly but shouldn’t
- **`PostStartHook` failure** — post-start lifecycle hook failing silently, container appears running but isn’t ready
- **Long pod startup times** — slow image pull, slow volume attach, readiness probe delays
- **Sidecar container not starting before main container** — init container vs sidecar ordering issues (K8s 1.29+ native sidecar)
- **Pods not getting scheduled after node added** — node labels missing, taint not tolerated by existing pods

### Node Health
- **Node `NotReady`** — kubelet logs, containerd status, disk/memory pressure conditions
- **Node disk pressure** — containerd image/log bloat, prune unused images with `ctr images rm`, journal log size
- **Node unschedulable** — taints, cordoned nodes, resource exhaustion
- **Kubelet not starting** — swap enabled, wrong cgroup driver (cgroupfs vs systemd), certificate issues
- **High inotify/file descriptor usage** — too many pods watching files, kernel limit exhaustion
- **Node time drift** — NTP desync causing certificate validation failures, JWT token rejections
- **Node kernel OOM kill** — system-level OOM vs container OOM, check `/var/log/messages` or `dmesg`
- **Node flapping between Ready/NotReady** — network instability, kubelet heartbeat timeout
- **CPU steal time on EC2** — noisy neighbor, instance type under-provisioned, check `iostat` / CloudWatch
- **Containerd socket unresponsive** — restart containerd, check for zombie shim processes
- **Node eviction storm** — soft vs hard eviction thresholds, all pods evicted simultaneously
- **Bottlerocket automatic update reboot** — nodes rebooting unexpectedly, check `uptime` and system logs via apiclient
- **Instance metadata service (IMDS) unreachable** — pods losing AWS IAM role credentials, IRSA token refresh failing

### Networking
- **Pod-to-pod connectivity** — test with netshoot pod, check CNI logs, iptables rules
- **DNS resolution failures** — CoreDNS pod logs, `nslookup` from inside pods, ndots/search domain misconfiguration
- **Ingress not routing** — ingress controller logs, backend service endpoints, TLS cert issues
- **Network Policy blocking traffic** — test with/without policies, trace with Hubble/Cilium
- **Service ClusterIP not reachable** — kube-proxy mode (iptables vs ipvs), stale endpoints
- **NodePort not accessible externally** — security group rules on EC2, firewall, `externalTrafficPolicy`
- **LoadBalancer service stuck in `Pending`** — AWS LB controller not installed, IAM permissions, subnet tags missing
- **Intermittent connection resets** — conntrack table full, TCP keepalive tuning, MTU mismatch (VxLAN overhead)
- **Pod IP conflicts** — CNI IP allocation exhausted, IPAM corruption, subnet too small
- **Service endpoints not updating** — kube-proxy lag, readiness probe misconfigured, endpointslice controller stuck
- **Cross-namespace DNS not working** — missing FQDN (`service.namespace.svc.cluster.local`), NetworkPolicy blocking port 53
- **AWS VPC CNI IP exhaustion** — node running out of secondary IPs, prefix delegation not enabled
- **HostNetwork pod conflicts** — port collision with another pod or host process
- **Egress traffic blocked** — missing NAT gateway route, security group egress rules, corporate proxy required

### Storage
- **PVC stuck in `Pending`** — StorageClass, provisioner logs, node affinity mismatch, zone mismatch
- **PVC stuck in `Terminating`** — remove `kubernetes.io/pvc-protection` finalizer
- **Volume mount failures** — permissions, fsGroup, SELinux context
- **`Multi-Attach error`** — RWO volume attached to multiple nodes, pod not fully terminated on old node
- **EBS volume stuck detaching** — force detach via AWS console, stale attachment entry in EC2
- **EFS mount timeout** — security group missing port 2049, VPC DNS not resolving EFS endpoint
- **StatefulSet pod not rescheduling** — PVC zone mismatch with new node, volume topology constraints
- **`fsck` on volume causing slow startup** — unclean unmount, `fsGroup` chown taking too long on large volumes
- **Ephemeral storage eviction** — container writing logs/temp files to root filesystem, hitting ephemeral limit
- **CSI driver not installed** — missing DaemonSet, CSI node plugin crashlooping
- **Snapshot restore failing** — VolumeSnapshot class mismatch, snapshot not in `ReadyToUse` state
- **ConfigMap/Secret changes not reflected in pod** — subPath mounts don’t hot-reload, pod restart required

### Debugging Without a Shell
- **No shell in distroless/scratch container** — `kubectl debug` ephemeral containers sharing process namespace
- **Copy pod with debug image** — `kubectl debug --copy-to` creates a modified clone of the pod
- **Attach to running process namespace** — inspect `/proc/<pid>/` for open files, env, fd, maps
- **Network debug from pod’s network namespace** — run `nsenter` on the node to enter pod network ns
- **Read pod filesystem without exec** — mount pod’s container overlay filesystem directly on node
- **Debug with node-level privileges** — `kubectl debug node/<node-name> -it --image=ubuntu`
- **Capture tcpdump from inside pod** — ephemeral container with `tcpdump`, or nsenter on node
- **Profile CPU/memory of a running Go/Java process** — attach async profiler, `pprof` endpoint, `jstack`
- **Inspect environment variables of a running container** — `cat /proc/1/environ | tr ‘\0’ ‘\n’` from node or ephemeral container

### RBAC & Auth
- **`Forbidden` errors** — `kubectl auth can-i`, check ServiceAccount bindings
- **`kubectl` token expiry** — re-authenticate, refresh kubeconfig
- **IRSA (IAM Roles for Service Accounts) not working** — missing OIDC provider, wrong annotation on SA, token audience mismatch
- **ServiceAccount token not mounted** — `automountServiceAccountToken: false`, missing projected volume
- **Webhook admission blocking resources** — identify which webhook is rejecting, check `ValidatingWebhookConfiguration`
- **OPA/Gatekeeper policy violations** — constraint template logs, dry-run mode to test before enforcing
- **Pod Security Admission violations** — privileged container blocked by namespace policy label
- **Audit log: who deleted a resource** — parse audit logs by verb=delete and resource kind
- **Cross-cluster access with kubeconfig** — context switching, merging kubeconfigs, expired certs in config
- **Node authorization issues** — kubelet certificate not approved (CSR pending), node can’t register

### Resource & Scaling
- **HPA not scaling** — metrics-server availability, custom metrics adapter, stabilization window too long
- **ResourceQuota exhaustion** — `kubectl describe resourcequota`, identify over-consuming workload
- **Evicted pods** — eviction thresholds, `status.phase=Failed` + `reason=Evicted`
- **VPA conflicting with HPA** — both trying to set replicas/resources simultaneously
- **Cluster Autoscaler not scaling up** — unschedulable pods not triggering scale, node group max reached, pending ASG instance
- **Cluster Autoscaler not scaling down** — PodDisruptionBudget blocking drain, non-evictable system pods
- **LimitRange preventing pod creation** — default limits applied, container exceeds max allowed
- **Namespace stuck in `Terminating`** — finalizer on a custom resource with deleted CRD, manually patch to remove
- **Deployment rollout stuck** — `maxUnavailable=0` + `maxSurge=0`, readiness probe never passing, quota exceeded mid-rollout
- **DaemonSet pod not on new node** — node taint not tolerated, DaemonSet node selector too restrictive
- **CronJob not firing** — timezone misconfiguration, `startingDeadlineSeconds` too tight, suspended flag set
- **Job completions not tracked** — indexed job misconfiguration, pod failure policy miscounted

### Cluster-Level
- **etcd backup/restore** — snapshot, restore procedure, member list corruption
- **Certificate expiry** — `kubeadm certs check-expiration`, renew, kubelet client cert rotation
- **Audit log analysis** — trace deletions, privilege escalation, unexpected API calls
- **API server overload** — watch storm, too many informers, LIST requests without resource version
- **Webhook timeout causing cascading failures** — admission webhook slow/down, `timeoutSeconds` too high, `failurePolicy: Fail`
- **Coredns crash loop** — OOM, config syntax error, upstream DNS unreachable
- **Scheduler not placing pods** — scheduler pod down, custom scheduler misconfiguration, `schedulerName` mismatch
- **Controller manager not reconciling** — leader election stuck, multiple controller managers running
- **etcd slow disk causing API server timeouts** — IOPS exhausted, etcd compaction not running, defragmentation needed
- **Kubernetes version skew** — kubectl/kubelet/apiserver version mismatch, deprecated API calls failing after upgrade
- **CRD deletion hanging** — custom resource instances not cleaned up, finalizer on CR instances blocking CRD removal
- **Mutating webhook modifying resources unexpectedly** — Istio/Linkerd sidecar injection, unexpected env injection, resource limits being overwritten
