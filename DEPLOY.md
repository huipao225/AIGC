# 灰袍 AIGC 检测 — 部署指南

## 准备工作

### 服务器要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| CPU | 2 核 | 4 核 |
| 内存 | 4 GB | 8 GB |
| 磁盘 | 20 GB | 40 GB |
| 系统 | Ubuntu 22.04 / CentOS 7+ | Ubuntu 22.04 LTS |
| 带宽 | 1 Mbps | 5 Mbps |

> 内存是关键：中文 RoBERTa (1.4GB) + DistilGPT-2 (350MB) 约需 2GB，加上系统开销，4GB 是最低要求。

### 你需要准备

1. **一台 VPS 云服务器**（阿里云 ECS / 腾讯云 CVM / AWS EC2 等），有公网 IP
2. **一个域名**（可选，用于 HTTPS）。没有域名可以直接用 IP 访问
3. **SSH 客户端**连接到服务器

---

## 部署步骤

### 第一步：连接服务器

```bash
ssh root@<你的服务器IP>
```

### 第二步：安装 Docker

```bash
# 官方一键安装脚本
curl -fsSL https://get.docker.com | sh

# 启动 Docker
systemctl enable docker
systemctl start docker

# 安装 Docker Compose
apt install -y docker-compose-plugin
# 或者: curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose && chmod +x /usr/local/bin/docker-compose
```

### 第三步：上传项目代码

**方式 A：Git 克隆（推荐）**

```bash
cd /opt
git clone <你的仓库地址> aigc
cd aigc
```

**方式 B：直接上传（如果没有 Git 仓库）**

在本地执行：
```bash
# 打包项目（排除模型缓存和 __pycache__）
cd D:\cctool\aigc
tar --exclude='models' --exclude='__pycache__' --exclude='*.pyc' -czf aigc.tar.gz backend nginx docker-compose.yml .env.example

# 上传到服务器
scp aigc.tar.gz root@<服务器IP>:/opt/
```

在服务器上：
```bash
cd /opt
tar -xzf aigc.tar.gz -C aigc/
cd aigc
```

### 第四步：配置环境

```bash
# 创建环境配置文件
cp .env.example .env

# 编辑配置（至少确认各项设置）
nano .env
```

关键配置项说明：

```bash
APP_NAME=灰袍 AIGC 检测
APP_VERSION=2.0.0
DEBUG=false                    # 生产环境必须为 false
LOG_LEVEL=INFO

# 模型配置
MODEL_PRIMARY=Hello-SimpleAI/chatgpt-detector-roberta-chinese
MODEL_CACHE_DIR=/app/models

# 文本限制
MAX_TEXT_LENGTH=50000          # 单次检测最大字符数，可调大
```

### 第五步：构建并启动

```bash
# 构建镜像并启动服务（首次约 10-20 分钟，需要下载依赖和模型）
docker compose up -d --build

# 查看构建进度
docker compose logs -f
```

> 国内服务器建议保持 Dockerfile 中的 `HF_ENDPOINT=https://hf-mirror.com` 配置，否则 HuggingFace 模型下载会失败。

### 第六步：验证部署

```bash
# 检查服务状态
docker compose ps

# 查看后端日志（确认模型加载完毕）
docker compose logs backend | grep "ready"

# 测试 API
curl http://localhost:8000/api/health
```

浏览器访问 `http://<服务器IP>` 即可看到检测页面。

---

## 域名与 HTTPS（推荐）

### 配置 DNS

将你的域名 A 记录指向服务器 IP：
```
aigc.yourdomain.com  →  <服务器IP>
```

### 修改 Nginx 配置

编辑 `nginx/nginx.conf`，将 `server_name _;` 改为你的域名：

```nginx
server {
    listen 80;
    server_name aigc.yourdomain.com;
    ...
}
```

### 申请 SSL 证书

```bash
# 安装 certbot
apt install -y certbot

# 先确保 nginx 在运行
docker compose restart nginx

# 申请证书（standalone 模式，需要先停掉 nginx 的 80 端口）
docker compose stop nginx
certbot certonly --standalone -d aigc.yourdomain.com

# 证书路径：
# /etc/letsencrypt/live/aigc.yourdomain.com/fullchain.pem
# /etc/letsencrypt/live/aigc.yourdomain.com/privkey.pem
```

挂载证书到 nginx 容器，更新 `docker-compose.yml`：

```yaml
nginx:
  volumes:
    - ./nginx/nginx.conf:/etc/nginx/conf.d/default.conf:ro
    - /etc/letsencrypt:/etc/letsencrypt:ro   # 添加这行
```

然后在 `nginx/nginx.conf` 中添加 443 端口监听：

```nginx
server {
    listen 80;
    server_name aigc.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name aigc.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/aigc.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/aigc.yourdomain.com/privkey.pem;

    # ... 其余配置同 80 端口
}
```

重新构建：
```bash
docker compose up -d --build
```

---

## 运维管理

### 常用命令

```bash
# 查看运行状态
docker compose ps

# 查看实时日志
docker compose logs -f backend

# 重启服务
docker compose restart

# 停止服务
docker compose down

# 更新代码后重新部署
git pull
docker compose up -d --build

# 清理旧镜像释放空间
docker system prune -a
```

### 性能调优

如果 CPU 推理太慢，可以增加 worker 数量（需要更多内存）：

编辑 `backend/Dockerfile` 最后一行：
```
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

### 安全建议

1. **配置防火墙**，只开放 80/443/22 端口：
   ```bash
   ufw allow 22
   ufw allow 80
   ufw allow 443
   ufw enable
   ```

2. **定期更新系统**：`apt update && apt upgrade -y`

3. **监控内存**：模型推理可能导致内存飙升，建议配置 swap：
   ```bash
   fallocate -l 4G /swapfile
   chmod 600 /swapfile
   mkswap /swapfile
   swapon /swapfile
   ```

---

## 故障排查

### 模型下载失败

如果构建时模型下载失败（国内常见问题），检查 `HF_ENDPOINT` 是否正确：

```bash
# 在服务器上测试镜像源
curl -I https://hf-mirror.com

# 手动设置环境变量
export HF_ENDPOINT=https://hf-mirror.com
```

### 内存不足 (OOM)

如果容器被 kill，说明内存不够：

```bash
# 查看内存使用
docker stats

# 增加 swap
# 或者限制同时处理的请求数
```

### 端口被占用

```bash
# 查看 80 端口占用
lsof -i :80

# 使用其他端口
# 修改 docker-compose.yml 中 nginx 的 ports 为 "8080:80"
```

---

## 预期性能

| 场景 | CPU 推理时间 | 内存占用 |
|------|------------|---------|
| 短文本 (<500字) | 200-400ms | ~2.5GB |
| 中文本 (500-2000字) | 400-800ms | ~3GB |
| 长文本 (2000-5000字) | 1-3s | ~3.5GB |

> 以上为 4 核 CPU 的参考数据。GPU 服务器可提速 5-10 倍。
