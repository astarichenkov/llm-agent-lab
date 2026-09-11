# Deployment Guide — clean Ubuntu VPS (Docker Compose)

This guide deploys the LLM Agent Lab to a fresh Ubuntu server using
Docker and Docker Compose. Target architecture:

```
Internet -> Nginx :80 -> Basic Auth -> rate limit -> FastAPI :8000 (inside Docker network)
```

The FastAPI port `8000` is **not** exposed publicly — only Nginx `:80` is.
`GET /health` is the only route **without** Basic Auth (kept public for
uptime monitoring).

> **`/api/compare` cost note:** one `POST /api/compare` request makes
> **two** DeepSeek provider calls (unrestricted + controlled). It is
> protected by the same Nginx Basic Auth and the same rate-limit zone as
> `/api/chat` (5 requests/min/IP shared), so worst-case provider spend per
> client is bounded.

---

## 1. Install Docker

```bash
# Update package index and install prerequisites
sudo apt update
sudo apt install -y ca-certificates curl

# Add Docker's official GPG key and repository
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine + CLI
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

# Add your user to the docker group (re-login afterwards)
sudo usermod -aG docker "$USER"
```

Log out and back in (or run `newgrp docker`) so the group takes effect.

## 2. Install the Docker Compose plugin

```bash
sudo apt install -y docker-compose-plugin
docker compose version   # expect: Docker Compose version v2.x
```

## 3. Clone the repository

```bash
git clone <your-repository-url> llm-agent-lab
cd llm-agent-lab
```

## 4. Set the DEEPSEEK_API_KEY securely

```bash
cp .env.example .env
# The key is stored ONLY on the server, in a git-ignored file:
#   DEEPSEEK_API_KEY=sk-your-real-key
nano .env
chmod 600 .env
```

Verify the file is ignored by git:

```bash
git status --short   # .env must NOT appear as untracked/changeable
git check-ignore .env && echo ".env is ignored"
```

## 5. Create the Nginx Basic Auth password file (REQUIRED)

The Nginx container mounts `./.htpasswd` (read-only) at
`/etc/nginx/.htpasswd` and refuses to start if the file is missing — the
stack **cannot start correctly until it is created**. The file is
git-ignored; never commit it.

```bash
sudo apt-get install -y apache2-utils
htpasswd -c .htpasswd student      # prompts for a strong password (twice)
chmod 600 .htpasswd
```

Verify it is ignored by git:

```bash
git check-ignore .htpasswd && echo ".htpasswd is ignored"
```

### Changing the password / adding / removing users

```bash
htpasswd .htpasswd student         # change password of user 'student'
htpasswd .htpasswd alice           # add another user
htpasswd -D .htpasswd alice        # remove a user
docker compose restart nginx       # apply after any change
```

### ⚠️ HTTPS warning — read before going live

HTTP Basic Auth only **Base64-encodes** `username:password`; it is **not
encrypted**. On a public VPS you **must** use **HTTPS** before relying on
real credentials, otherwise anyone sniffing port 80 can read them. Plan:

1. Point a domain (e.g. `app.example.com`) at the VPS.
2. Open port 80 to the public (already required).
3. Install Certbot and obtain a certificate:

   ```bash
   sudo apt install -y certbot python3-certbot-nginx
   sudo certbot --nginx -d app.example.com
   ```

4. Certbot rewrites `nginx/nginx.conf` to add TLS and redirect HTTP→HTTPS.
   Re-run `sudo certbot renew` automatically via its systemd timer.

Do **not** claim HTTP Basic Auth over plain HTTP is secure.

## 6. Start the containers

```bash
docker compose up -d --build
```

## 7. Check container status

```bash
docker compose ps
```

Both `app` and `nginx` should be `Up` and healthy.

## 8. Check logs

```bash
docker compose logs -f app     # FastAPI logs (Ctrl+C to exit)
docker compose logs -f nginx   # Nginx access/error logs
```

### Persistent logs (host disk)

In addition to `docker compose logs`, the stack writes persistent logs to
host directories that survive container recreation:

- Application: `./logs/app.log` (auto-rotated ~5 MB × 3 backups)
- Nginx access: `./logs/nginx/access.log`
- Nginx errors: `./logs/nginx/error.log`

They are bind-mounted (`./logs` -> `/app/logs`; `./logs/nginx` ->
`/var/log/nginx`) and are git-ignored. `docker compose` creates the host
directories automatically; on a Linux VPS ensure the `app` and `nginx`
users can write there (Docker handles this as root-owned files, which is
fine for reading them with `sudo`/`cat`). The application falls back to
stdout-only if the log file cannot be written, so the app never fails due
to logging.

### Persistent Agent context (Day 7)

The SQLite database with dialog history is bind-mounted from `./data` to
`/app/data` (`AGENT_DB_PATH=data/agents.db`). It survives
`docker compose restart` **and** container recreation. `./data` is
git-ignored. Back it up like any file:

```bash
cp data/agents.db data/agents.db.bak
```

To reset all agent context, remove the file while the app is stopped:

```bash
docker compose stop app && rm -f data/agents.db && docker compose start app
```

## 9. Test /health

```bash
curl -s http://localhost/health
# {"status":"ok","application":"llm-agent-lab"}
```

`/health` is public. Everything else requires Basic Auth:

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost/          # 401
curl -s -u student:PASSWORD http://localhost/ | head                 # 200
```

Test the real chat endpoint (this consumes a tiny amount of balance):

```bash
curl -s -u student:PASSWORD -X POST http://localhost/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Say OK"}'
```

Verify the rate limit is active (these requests are rejected by Nginx
before FastAPI/DeepSeek, so no balance is spent):

```bash
for i in $(seq 1 15); do
  curl -s -o /dev/null -w "%{http_code} " http://localhost/api/chat
  sleep 0.2
done
echo
# expect a mix of 401 and eventually 429
```

Open `http://<server-ip>/` in a browser and log in when prompted. If you
have a domain, point it at the server and enable HTTPS with Certbot (see
section 5).

## 10. Restarting

```bash
docker compose restart          # restart containers (no rebuild)
docker compose down             # stop and remove containers
docker compose up -d            # start again
```

## 11. Updating the application

```bash
cd llm-agent-lab
git pull                        # pull the latest code
docker compose up -d --build    # rebuild image + restart
docker compose ps               # verify
curl -s http://localhost/health
```

## 12. Rollback considerations

- Docker Compose keeps the previous image layers locally, but the `latest`
  tag is overwritten on rebuild. For reliable rollback, tag releases:

  ```bash
  docker compose build
  docker tag llm-agent-lab:latest llm-agent-lab:release-2025-01-01
  ```

  To roll back to a known-good tag:

  ```bash
  docker compose down
  # edit docker-compose.yml: image: llm-agent-lab:release-2025-01-01
  docker compose up -d
  ```

- Before upgrading, check the new commit for breaking changes
  (`git log --oneline -5`, `git diff HEAD~1`).
- The `.env` file survives `git pull` (it is git-ignored), so the API key
  is never lost or overwritten by updates.
- If a deployment breaks, stop the new containers and re-run the previous
  image: `docker compose down && docker compose up -d` (after restoring
  the previous tag in `docker-compose.yml`).
- Consider snapshotting the VPS (cloud provider snapshot / `doctl` /
  AWS AMI) before major upgrades.

---

## Troubleshooting

| Symptom                              | Likely fix                                   |
|--------------------------------------|----------------------------------------------|
| Nginx container exits on start       | `.htpasswd` missing — create it (section 5), then `docker compose up -d` |
| `401` from the browser / curl        | Wrong username/password in `.htpasswd`, or credentials not sent; use `curl -u student:PASSWORD` |
| `401` from `/api/chat`               | Wrong/expired `DEEPSEEK_API_KEY` in `.env`; `docker compose up -d` again |
| `429` from `/api/chat`               | Rate limit hit (5 req/min/IP) — wait a minute; do not raise the limit casually |
| `502` from `/api/chat`               | Network/firewall blocks outbound HTTPS, or DeepSeek is unreachable |
| Port 80 already in use               | `sudo lsof -i :80` to find the process; free the port |
| `docker compose` not found           | Install the compose plugin (section 2)       |
| Nginx 502 on `/health`               | `docker compose logs app`; app crashed?      |
