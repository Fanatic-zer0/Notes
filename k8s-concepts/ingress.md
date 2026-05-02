# Ingress & External Traffic in Kubernetes

## Overview

**Ingress** provides HTTP/HTTPS routing to services based on hostnames and paths. It's a Layer 7 (Application) load balancer, unlike Services which operate at Layer 4 (Transport).

## Why Ingress?

**Without Ingress:**
- Need separate LoadBalancer per service ($$$)
- No path-based routing
- No SSL termination at cluster edge
- No hostname routing

**With Ingress:**
- Single LoadBalancer for all services
- Path & hostname routing
- SSL/TLS termination
- Auth, rate limiting, rewrites

## Architecture

```
Internet
    │
    ├─→ Cloud LoadBalancer (1 public IP)
    │         203.0.113.100
    │
    ├─→ Ingress Controller (nginx/traefik/istio)
    │   Pods running on nodes: 
    │   - Watch Ingress resources
    │   - Configure routing rules
    │   - Handle SSL termination
    │
    ├─→ Service (ClusterIP)
    │   - frontend-service:80
    │   - api-service:8080
    │
    └─→ Backend Pods
        - frontend-pod-1, frontend-pod-2
        - api-pod-1, api-pod-2, api-pod-3
```

## Complete Traffic Flow

### Request Flow
```
User Browser: https://api.example.com/users
     │
     ├─→ 1. DNS Resolution
     │       api.example.com → 203.0.113.100 (LoadBalancer IP)
     │
     ├─→ 2. HTTPS request to LoadBalancer
     │       Host: api.example.com
     │       Path: /users
     │
     ├─→ 3. LoadBalancer forwards to Ingress Controller
     │       NodePort or direct pod routing
     │       → nginx-ingress-controller:443
     │
     ├─→ 4. Ingress Controller (nginx pod)
     │       ┌──────────────────────────────────┐
     │       │ SSL Termination (TLS cert)       │
     │       │ Decrypt HTTPS → HTTP             │
     │       └──────────────────────────────────┘
     │       ┌──────────────────────────────────┐
     │       │ Routing Rules Evaluation         │
     │       │ Host: api.example.com            │
     │       │ Path: /users                     │
     │       │ Match: Ingress rule #2           │
     │       │ Backend: api-service:8080        │
     │       └──────────────────────────────────┘
     │
     ├─→ 5. Forward to Service (ClusterIP)
     │       HTTP request to api-service:8080
     │       Internal cluster call
     │
     ├─→ 6. kube-proxy load balancing
     │       api-service (10.96.20.30:8080)
     │       → random pod: 10.244.2.15:8080
     │
     ├─→ 7. Backend Pod processes request
     │       api-pod-2 handles GET /users
     │       Returns JSON response
     │
     └─→ 8. Response path (reverse)
              Pod → Service → Ingress → LB → User
              Ingress re-encrypts if needed
```

## Ingress Resource Example

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: app-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.example.com
    - www.example.com
    secretName: example-tls  # TLS certificate
  rules:
  # Rule 1: Frontend
  - host: www.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-service
            port:
              number: 80
  
  # Rule 2: API
  - host: api.example.com
    http:
      paths:
      - path: /users
        pathType: Prefix
        backend:
          service:
            name: user-service
            port:
              number: 8080
      - path: /orders
        pathType: Prefix
        backend:
          service:
            name: order-service
            port:
              number: 8080
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-gateway
            port:
              number: 8080
```

## Ingress Controllers

Ingress is just a **specification**. You need an **Ingress Controller** to implement it.

### Popular Controllers

| Controller | Maintainer | Features | Use Case |
|------------|------------|----------|----------|
| **ingress-nginx** | Kubernetes community | Full-featured, widely adopted OSS | General purpose (most common) |
| **nginx-ingress** | NGINX Inc | Enterprise NGINX Plus features | Commercial/Enterprise |
| **Traefik** | Traefik Labs | Auto-discovery, dashboard | Microservices |
| **HAProxy** | HAProxy | High performance | High traffic |
| **Istio Gateway** | Istio | Service mesh integration | Advanced routing |
| **AWS ALB** | AWS | Native AWS integration | EKS clusters |
| **Contour** | VMware | Envoy-based | Cloud-native |

> **ingress-nginx vs nginx-ingress:** These are two distinct controllers. `ingress-nginx` (registry: `registry.k8s.io/ingress-nginx/controller`) is the Kubernetes community project and the most widely deployed. `nginx-ingress` (Nginx Inc) is the commercial/enterprise offering. They share similar annotation syntax but differ in defaults and paid features.

### NGINX Ingress Controller Deployment

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ingress-nginx-controller
  namespace: ingress-nginx
spec:
  type: LoadBalancer  # Creates cloud LB
  ports:
  - name: http
    port: 80
    targetPort: 80
  - name: https
    port: 443
    targetPort: 443
  selector:
    app.kubernetes.io/name: ingress-nginx
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ingress-nginx-controller
  namespace: ingress-nginx
spec:
  replicas: 3  # HA setup
  selector:
    matchLabels:
      app.kubernetes.io/name: ingress-nginx
  template:
    metadata:
      labels:
        app.kubernetes.io/name: ingress-nginx
    spec:
      containers:
      - name: controller
        image: registry.k8s.io/ingress-nginx/controller:v1.8.0
        args:
        - /nginx-ingress-controller
        - --election-id=ingress-controller-leader
        - --controller-class=k8s.io/ingress-nginx
        ports:
        - name: http
          containerPort: 80
        - name: https
          containerPort: 443
```

## How Ingress Controller Works

### Internal Architecture
```
┌────────────────────────────────────────────────────────────────┐
│ Ingress Controller Pod                                         │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ Controller Process                                        │ │
│  │ - Watches Ingress, Service, Endpoint resources           │ │
│  │ - Watches Secret (for TLS certs)                         │ │
│  │ - Generates nginx.conf from Ingress rules                │ │
│  │ - Reloads nginx when config changes                      │ │
│  └────────────────┬─────────────────────────────────────────┘ │
│                   │ Generates config                          │
│                   ↓                                            │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ NGINX Process                                            │ │
│  │ - Listens on :80 and :443                                │ │
│  │ - SSL termination                                        │ │
│  │ - Routing based on Host/Path                             │ │
│  │ - Proxies to Service ClusterIPs                          │ │
│  │                                                           │ │
│  │ Generated nginx.conf:                                    │ │
│  │   server {                                               │ │
│  │     listen 443 ssl;                                      │ │
│  │     server_name api.example.com;                         │ │
│  │     location /users {                                    │ │
│  │       proxy_pass http://user-service.default.svc:8080;   │ │
│  │     }                                                     │ │
│  │   }                                                       │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

### Watch and Update Flow
```
kubectl apply -f ingress.yaml
     │
     ├─→ 1. kube-apiserver stores Ingress resource in etcd
     │
     ├─→ 2. Ingress Controller watches Ingress events
     │         WATCH /apis/networking.k8s.io/v1/ingresses
     │
     ├─→ 3. Controller receives ADD/UPDATE event
     │         Event: Ingress "app-ingress" added
     │         Rules: api.example.com → api-service:8080
     │
     ├─→ 4. Controller queries related resources
     │         - Service: api-service (get ClusterIP)
     │         - Endpoints: api-service (get pod IPs)
     │         - Secret: example-tls (get TLS cert)
     │
     ├─→ 5. Generate NGINX configuration
     │         Create nginx.conf from all Ingress rules
     │
     ├─→ 6. Validate configuration
     │         nginx -t -c /tmp/nginx.conf
     │
     ├─→ 7. Reload NGINX
     │         nginx -s reload
     │         Zero-downtime reload (~100ms)
     │
     └─→ 8. Ready to route traffic
```

## Path Types

### Prefix (Most Common)
```yaml
pathType: Prefix
path: /api
```
**Matches:**
- /api
- /api/
- /api/users
- /api/users/123

### Exact
```yaml
pathType: Exact
path: /api
```
**Matches:**
- /api

**Does NOT match:**
- /api/
- /api/users

### ImplementationSpecific
```yaml
pathType: ImplementationSpecific
path: /api/*
```
Controller-specific behavior (NGINX uses regex).

## SSL/TLS Termination

### Automatic HTTPS with cert-manager

```yaml
# Install cert-manager first
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Create ClusterIssuer for Let's Encrypt
---
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
---
# Ingress with auto TLS
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: app-ingress
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.example.com
    secretName: api-tls  # cert-manager creates this automatically
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 8080
```

**How cert-manager works:**
1. Reads Ingress with `cert-manager.io/cluster-issuer` annotation
2. Creates ACME challenge (HTTP-01 or DNS-01)
3. Requests certificate from Let's Encrypt
4. Stores cert in Secret `api-tls`
5. Ingress Controller uses this Secret for SSL termination
6. Auto-renews before expiry (every 60 days)

## Advanced Ingress Features

### Rewrite Rules
```yaml
metadata:
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /$2
spec:
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /api(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 8080
```
**Effect:** `/api/users` → forwards as `/users` to backend

### Rate Limiting
```yaml
metadata:
  annotations:
    nginx.ingress.kubernetes.io/limit-rps: "10"
    nginx.ingress.kubernetes.io/limit-connections: "5"
```

### Basic Authentication
```bash
# Create htpasswd secret
htpasswd -c auth admin
kubectl create secret generic basic-auth --from-file=auth
```

```yaml
metadata:
  annotations:
    nginx.ingress.kubernetes.io/auth-type: basic
    nginx.ingress.kubernetes.io/auth-secret: basic-auth
    nginx.ingress.kubernetes.io/auth-realm: 'Authentication Required'
```

### CORS Headers
```yaml
metadata:
  annotations:
    nginx.ingress.kubernetes.io/enable-cors: "true"
    nginx.ingress.kubernetes.io/cors-allow-origin: "https://app.example.com"
    nginx.ingress.kubernetes.io/cors-allow-methods: "GET, POST, PUT, DELETE"
```

### Canary Deployments
```yaml
# Production ingress
metadata:
  name: app-ingress-prod
spec:
  rules:
  - host: app.example.com
    http:
      paths:
      - path: /
        backend:
          service:
            name: app-v1
            port:
              number: 80
---
# Canary ingress (10% traffic)
metadata:
  name: app-ingress-canary
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    nginx.ingress.kubernetes.io/canary-weight: "10"
spec:
  rules:
  - host: app.example.com
    http:
      paths:
      - path: /
        backend:
          service:
            name: app-v2  # New version
            port:
              number: 80
```

## Multi-Region Ingress (Global Load Balancing)

### With Cloud Providers

**AWS:**
- Use Route53 for DNS-based routing
- ALB Ingress Controller per region
- Health checks to active regions

**GCP:**
- Global HTTP(S) Load Balancer
- Multi-cluster ingress
- Anycast IP routing

**Azure:**
- Traffic Manager for global routing
- Application Gateway per region

## Troubleshooting

```bash
# Check ingress status
kubectl get ingress
kubectl describe ingress app-ingress

# View ingress controller logs
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller

# Test DNS resolution
nslookup api.example.com

# Check TLS certificate
openssl s_client -connect api.example.com:443 -servername api.example.com

# View generated NGINX config
kubectl exec -n ingress-nginx ingress-nginx-controller-xxx -- cat /etc/nginx/nginx.conf

# Test backend connectivity from ingress pod
kubectl exec -n ingress-nginx ingress-nginx-controller-xxx -- curl http://api-service.default.svc:8080
```

## Performance Best Practices

1. **Use Connection Pooling** - Ingress maintains persistent connections to backends
2. **Enable HTTP/2** - Better performance for modern clients
3. **Configure Buffer Sizes** - Tune for your payload sizes
4. **Set Appropriate Timeouts** - Avoid holding connections too long
5. **Use Keep-Alive** - Reduce connection overhead
6. **Enable Compression** - Reduce bandwidth usage

```yaml
metadata:
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "60"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "60"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "60"
    nginx.ingress.kubernetes.io/enable-compression: "true"
```

---

**Next:** [DNS Resolution](04-dns-resolution.md)
