# Publishing the model packs to HuggingFace

One-time, from a machine holding the full `models/` directory.

## 1. Prerequisites

```bat
uv tool install huggingface_hub[cli]
hf auth login
```

## 2. Confirm LFS tracking

The repository already has a `.gitattributes`. Confirm it covers both patterns —
a 375 MB file committed outside LFS is rejected:

```
*.onnx filter=lfs diff=lfs merge=lfs -text
*.engine filter=lfs diff=lfs merge=lfs -text
```

## 3. Move the existing engines under a scoped path

The 21 `.engine`/`.profile` files are currently at the repository root. Move
them so the ONNX packs are not buried among them.

**No git clone is needed** — a HuggingFace repo is a git repo underneath, but
the HTTP API does all of this without one, and a clone would mean pulling 3.1 GB
down only to push it back up.

Re-upload the engines under the new prefix, then delete the originals:

```bat
hf upload Symphoenix/sylc_TRT G:\SyLC-main\trt_cache trt/sm89-trt10.16.1.11 ^
    --include "TensorrtExecutionProvider_*"
```

(If the local `trt_cache/` no longer holds them, download them first with
`hf download Symphoenix/sylc_TRT --local-dir <tmp>`.)

Then remove the root-level copies:

```python
from huggingface_hub import HfApi
api = HfApi()
root = [f for f in api.list_repo_files("Symphoenix/sylc_TRT")
        if f.startswith("TensorrtExecutionProvider_")]
api.delete_files("Symphoenix/sylc_TRT", root,
                 commit_message="Scope the sm89 engine cache under trt/")
print(f"removed {len(root)} root-level engine files")
```

## 4. Upload the model packs — staged from the manifest, never globbed

**Do not point `hf upload` at `models/` with a pattern.** That directory holds
development artifacts beside the twenty shipping graphs, and the obvious
patterns match them: `da3_base_*.onnx` catches `da3_base_756_t3.onnx` (394 MB)
and `da3_base_756_fp16.onnx` (198 MB), `da3_small_*.onnx` catches
`da3_small_518_clean.onnx` (101 MB). That is 693 MB of exports the "Never
upload" list below forbids — published by the very command meant to respect it.

Adding `--exclude` rules would fix those three and leak the fourth. Instead
drive the upload from `models/MANIFEST.json`, which already names exactly what
ships **and is the same list every user's SHA-256 check is verified against**.
Anything that can disagree with the manifest eventually will.

Stage first. This hard-links rather than copies (same volume, so it is instant
and costs no disk), and it re-checks every size against the manifest so a stale
manifest is caught before anything is published, not after:

```bat
G:\SyLC-main\.venv\Scripts\python.exe G:\SyLC-main\tools_dev\stage_hf_upload.py
```

**Check the last line before going further.** It reads:

```
staged into G:\SyLC-main\.hf_upload_stage (20 hard-linked, 0 copied)
```

**The gate is the count: it must total 20 files** — ten per pack. That is the
whole point, and it is what the manifest check is measured against.

The hard-linked/copied split is **informational, not a gate**. Hard links need
`models/` and the repo root on the same volume; when they are not, `stage_hf_upload.py`
falls back to `shutil.copy2` and a perfectly correct run prints
`(0 hard-linked, 20 copied)` — slower and it costs the disk space, but the
staged bytes are identical. Any split summing to 20 is a pass.

The script stages atomically, so a failure leaves no `.hf_upload_stage\` at all
rather than a half-filled one; if the total is anything other than 20, **stop
and do not run the uploads below**, because the next two commands publish
whatever those directories hold.

Then upload the two staged directories, which contain nothing else:

```bat
hf upload Symphoenix/sylc_TRT G:\SyLC-main\.hf_upload_stage\small onnx/small
hf upload Symphoenix/sylc_TRT G:\SyLC-main\.hf_upload_stage\base  onnx/base
```

Delete `G:\SyLC-main\.hf_upload_stage\` afterwards.

**Never upload** `models/DA3-GIANT/`, `models/DA3-LARGE`, `da3_base_756_t3.onnx`,
`da3_base_756_fp16.onnx`, `da3_small_518_clean.onnx`, `da3_small.onnx`, or any
`.bak_*` file. The first is CC-BY-NC-4.0 and must never be redistributed; the
rest are development artifacts. The staging step above enforces this
structurally — it copies only what the manifest names — but the list stays here
because it is also the rule for any future manual upload.

## 5. Upload the two README files

`tools_dev/hf_repo_docs/README.md` → repository root `README.md`
`tools_dev/hf_repo_docs/trt_README.md` → `trt/sm89-trt10.16.1.11/README.md`

## 6. Report the commit SHA

```bat
curl -s https://huggingface.co/api/models/Symphoenix/sylc_TRT | ^
    python -c "import json,sys; print(json.load(sys.stdin)['sha'])"
```

Hand that SHA back — it gets pinned into `models/MANIFEST.json` and
`tools_dev/setup_tensorrt.py` in the next step. **Nothing works until it is
pinned**: both files currently carry the literal `PENDING_UPLOAD`.
