# Private cloud-to-Windows artifact bridge

This prepares a write-only route into `C:\Users\rruhl\Local\Projects\Fly-Connectome\checkpoints`. Repository code still travels through Git; learned checkpoint files travel through this bridge. The receiver needs only Python's standard library and listens on **127.0.0.1:8769**, not the LAN. Tailscale Serve supplies the private HTTPS endpoint. Do **not** use Tailscale Funnel.

The bridge is not connected merely because these scripts exist. The current cloud environment has no configured VPN route and restricted outbound hosts. You must connect that environment to the same Tailnet, allow the exact private HTTPS hostname, and configure its secret before an actual upload can succeed. Local tests do not establish a cloud-to-Windows connection.

## 1. Update the Windows checkout and create the local token

In PowerShell:

```powershell
$Repo = 'C:\Users\rruhl\Local\Projects\Fly-Connectome'
Set-Location $Repo
git fetch origin
git switch reconstruction/m1b-continuation
git pull --ff-only
py -3 scripts\artifact_bridge.py init-token
$TokenFile = Join-Path $env:LOCALAPPDATA 'FlyConnectomeBridge\token'
```

`init-token` creates 256 random bits, or reuses an existing valid token. It prints **only the file location**, never the token. The default location is outside Git. Keep this file private. `--token-file PATH` can select a different private location; do not put it in the repository.

## 2. Start the receiver and keep its terminal open

```powershell
py -3 scripts\artifact_bridge.py receive --root "$Repo\checkpoints" --token-file "$TokenFile"
```

No NumPy, Torch, SSH server, router forwarding or inbound Windows firewall rule is needed for this loopback receiver. It does not run the viewer or training. You can check it from a second PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:8769/health
```

The only unauthenticated information is `ready: true`.

## 3. Expose that loopback port privately with Tailscale

In the second PowerShell window, sign in to Tailscale if needed. This also handles installations where its CLI is not on PATH:

```powershell
$TailscaleCommand = Get-Command tailscale -ErrorAction SilentlyContinue
$TailscaleExe = if ($TailscaleCommand) { $TailscaleCommand.Source } else { "$env:ProgramFiles\Tailscale\tailscale.exe" }
& $TailscaleExe status
& $TailscaleExe serve status
```

Check the existing Serve configuration first. If HTTPS port 443 is unused, add this private route:

```powershell
& $TailscaleExe serve --bg http://127.0.0.1:8769
& $TailscaleExe serve status
```

If another service already uses that route, do not overwrite it. Choose an unused HTTPS port instead, for example `& $TailscaleExe serve --bg --https=8443 http://127.0.0.1:8769`, then use the resulting URL **including that port** and allow the same port in the cloud environment.

Use the actual `https://...ts.net` URL reported by **serve status**. Do not invent a hostname or keep a trailing DNS dot. If Tailscale asks you to enable HTTPS, follow its account-specific instructions. Serve is private to the Tailnet and its access policy; Funnel is public and is not part of this setup.

The private bridge URL receives files. The playable viewer uses its separate local port, normally 8767.

## 4. Configure the cloud environment, without sharing the token in chat

Use this environment's VPN/network/secret configuration to:

1. Connect it to the **same Tailnet** as the Windows laptop.
2. Permit HTTPS to the **exact hostname** from `tailscale serve status` (normally port 443).
3. Set ordinary runtime variable **`PONG_BRIDGE_URL`** to that private base HTTPS URL.
4. Set secret **`PONG_BRIDGE_TOKEN`** to the local token file's value.

You can copy the local value directly to the Windows clipboard without printing it:

```powershell
Get-Content "$env:LOCALAPPDATA\FlyConnectomeBridge\token" -Raw | Set-Clipboard
```

Paste only into the environment's secret field. Never paste it into chat, Git, a URL or a command-line token argument. Afterward, clear the clipboard if desired with `Set-Clipboard -Value ''`. The client accepts the runtime's opaque secret/proxy representation and uses inherited proxy settings; it does not bypass the proxy or disable certificate verification. Configuring Tailscale on Windows alone does not configure the cloud side. The assistant cannot claim or complete a connection while that environment configuration remains unavailable.

## 5. Upload a file from the cloud

Once the route and secret are configured, from the cloud repository root:

```sh
python scripts/artifact_bridge.py send \
  --file /workspace/m1b-experiments/pong-interception-checkpoint-20261025/weights.npy \
  --path pong-interception-checkpoint-20261025/weights.npy
```

The client reads the URL and token from the configured environment, streams the file with an explicit length and SHA256, verifies HTTPS normally, and checks the receiver's acknowledgement. Success prints only the destination-relative name, byte count and hash. A private token file can be used instead of the environment with `--token-file PATH`; there is deliberately no token-value argument.

Transfer the **prepared viewer checkpoint tree**, preserving its relative paths, to retain all five checkpoint choices. The final bundle alone also needs its scaling, visual prior, manifest and other supplied files; uploading only `weights.npy` is not a complete viewer installation. Earlier checkpoint folders are separate from the final bundle. Do not recursively upload the entire research experiment directory or its journals. The receiver never unpacks archives or executes received scripts.

For a prepared, bounded checkpoint directory, a cloud-side loop can use the same client without exposing credentials:

```python
from pathlib import Path
import os
from scripts.artifact_bridge import upload

prepared = Path('/path/to/prepared/viewer/checkpoints')  # The curated tree, not all experiments.
for artifact in sorted(prepared.rglob('*')):
    if artifact.is_symlink():
        raise ValueError('Do not upload symlinks')
    if artifact.is_file():
        relative = artifact.relative_to(prepared).as_posix()
        receipt = upload(os.environ['PONG_BRIDGE_URL'], artifact, relative,
                         os.environ['PONG_BRIDGE_TOKEN'])
        print(relative, receipt['bytes'], receipt['sha256'])
```

The receiver's default root puts the resulting files under the repository's ignored `checkpoints` directory. Once the complete tree is present, use the project's existing Python environment and [viewer launch instructions](frozen-pong.md), or its portable launcher:

```powershell
Set-Location 'C:\Users\rruhl\Local\Projects\Fly-Connectome'
py -3 scripts\run_pong_viewer.py --checkpoints .\checkpoints
```

The viewer itself needs the project's NumPy/Torch dependencies; the bridge does not.

## Existing files and limits

Uploads never silently overwrite a file. To intentionally replace one, first obtain its current hash locally, for example:

```powershell
(Get-FileHash "$Repo\checkpoints\pong-interception-checkpoint-20261025\weights.npy" -Algorithm SHA256).Hash
```

Then the cloud caller must explicitly pass that nonsecret hash:

```sh
python scripts/artifact_bridge.py send --file /path/to/new/weights.npy \
  --path pong-interception-checkpoint-20261025/weights.npy \
  --expected-existing-sha256 EXISTING_64_HEX_DIGIT_HASH
```

The receiver checks the old hash again immediately before atomic replacement. A missing or changed expected target is rejected. Hash mismatches, incomplete transfers and refused overwrites leave the existing file untouched. Each upload is limited to 64 MiB and a 20-second body-read deadline. The receiver uses bounded `read1` calls with the remaining absolute deadline, rather than allowing a slow sender to renew the timeout indefinitely. Only portable relative filenames are accepted: no parent traversal, absolute paths, Windows drives, backslashes, alternate data streams, reserved device names, symlinks or directory reparse points. There are no file-read, listing, shell, extraction or delete endpoints. Redirects are refused by the client so a bearer credential is never forwarded to another URL.

Stop the receiver with Ctrl+C when finished. To disable this Serve mapping, use `& $TailscaleExe serve --https=443 off` from PowerShell (substitute your chosen HTTPS port if different). The private token file remains available for reuse; no daemon, scheduled task or automatic synchronization is installed.

## Local verification

```sh
python -m pytest tests/test_artifact_bridge.py -q
```

These tests use an ephemeral loopback receiver and a fake token. They check actual streamed upload, wrong authorization, traversal/Windows path aliases, symlink refusal, size/length/deadline handling, digest cleanup, compare-and-swap overwrite, private token-file setup, bounded client reads, HTTPS requirements and redirect refusal. They do not contact the laptop or use a real token.
