# modal_app.py — Free Video Generation Studio backend on Modal
#
# What it is: a public web API that runs Wan 2.1 (1.3B, open weights) on a
# Modal T4 GPU and returns an .mp4. The site calls it when the visitor
# fills in the "Modal server URL" field, so visitors need NO key and NO
# Hugging Face account. Cost: T4 is $0.59/h; Modal's free Starter tier is
# $30/month of credits (~50 T4-hours), no separate video API bill.
#
# TEXT-TO-VIDEO ONLY (2026-11-23): the Wan 2.1 I2V checkpoint
# (Wan-AI/Wan2.1-I2V-1.3B-480P) is a GATED Hugging Face repo — anonymous
# download fails with 401 "Invalid username or password", and the only
# mirrors are 14B checkpoints (too big for a 16GB T4) or GGUF files that
# diffusers cannot load. The T2V checkpoint is NOT gated and downloads
# fine, so this server runs T2V only. Image-to-video requests get a clear
# error and the site falls back to the HF Spaces engines. To enable I2V
# later: create a Hugging Face account, accept the Wan 2.1 licence, make a
# read token, store it as a Modal Secret "hf-token", and restore the I2V
# download + pipeline below.
#
# Limits (honest): one call = one clip of at most 5 seconds (Wan 2.1 1.3B
# caps at 81 frames @16fps). Longer videos come from the site's "long
# video" mode, which chains several 5s calls on last frames. A web request
# is capped at 150s, so generation runs async: POST /generate returns a
# job_id immediately, the site polls GET /status/{job_id}, then downloads
# GET /download/{job_id}. Cold start is 1-3 min the first time (model
# already baked into the image at build time).
#
# Deploy (needs a free Modal account — sign up at modal.com with GitHub):
#   pip install modal
#   modal deploy modal_app.py
#   # copy the printed web URL (https://<you>--video-studio-api.modal.run)
#   # into the site's "Modal server URL" field.
#
# Optional key gate (so strangers can't drain the $30): create a Modal
# Secret named "video-studio-key" with VIDEO_KEY=<a secret string>, then
# add secrets=[modal.Secret.from_name("video-studio-key")] to the api
# function below. The site sends "key" in the /generate JSON; a wrong or
# missing key gets 401. Without the secret attached, the endpoint is open.

import os
import uuid

import modal

APP_NAME = "video-studio"
T2V_MODEL = "Wan-AI/Wan2.1-T2V-1.3B"
MAX_FRAMES = 81  # Wan 2.1 1.3B cap: 81 frames = ~5s at 16fps
FPS = 16


def _download_models():
    from huggingface_hub import snapshot_download

    snapshot_download(T2V_MODEL)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "diffusers>=0.32.0",
        "transformers>=4.48.0",
        "accelerate",
        "huggingface_hub",
        "imageio-ffmpeg",
        "fastapi",
        "pillow",
    )
    .run_function(_download_models, timeout=1800)
)

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("video-studio-output", create_if_missing=True)
jobs = modal.Dict.from_name("video-studio-jobs", create_if_missing=True)


@app.function(image=image, gpu="T4", timeout=1800, volumes={"/out": vol})
def generate_clip(job_id, prompt, negative_prompt, width, height,
                  duration_seconds, steps, guidance, seed, image_b64):
    import torch
    from diffusers import WanPipeline
    from diffusers.utils import export_to_video

    jobs[job_id] = {"status": "running"}
    try:
        # Wan frames must be 4n+1 and at most 81.
        want = int(round(float(duration_seconds) * FPS))
        frames = max(9, min(MAX_FRAMES, want // 4 * 4 + 1))
        # The diffusers Wan pipeline uses CFG ~5; the site's CausVid-style
        # guidance of 1.0 would barely steer it, so floor it at 5.0.
        cfg = float(guidance) if guidance and float(guidance) >= 2 else 5.0
        gen = (
            torch.Generator("cuda").manual_seed(int(seed))
            if seed is not None and int(seed) >= 0
            else None
        )
        if image_b64:
            # T2V-only server: the I2V checkpoint is gated (see header).
            # Raising here makes the site fall back to the next engine.
            raise RuntimeError(
                "image-to-video is not available on this server "
                "(Wan 2.1 I2V is a gated Hugging Face repo); "
                "the site will try the next engine"
            )
        pipe = WanPipeline.from_pretrained(T2V_MODEL, torch_dtype=torch.bfloat16)
        pipe.to("cuda")
        out = pipe(
            prompt=prompt, negative_prompt=negative_prompt,
            height=int(height), width=int(width), num_frames=frames,
            guidance_scale=cfg, num_inference_steps=int(steps),
            generator=gen,
        ).frames[0]
        path = f"/out/{job_id}.mp4"
        export_to_video(out, path, fps=FPS)
        vol.commit()
        jobs[job_id] = {"status": "done"}
    except Exception as e:  # surfaced verbatim to the site's status log
        jobs[job_id] = {"status": "error", "error": str(e)[:500]}


@app.function(image=image, volumes={"/out": vol})
@modal.asgi_app()
def api():
    import fastapi
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware

    web = fastapi.FastAPI()
    # The site runs on GitHub Pages (a different origin), so the browser
    # needs CORS headers or every fetch fails before it even starts.
    web.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web.get("/")
    def root():
        return {"ok": True, "service": "video-studio",
                "model": "Wan2.1-1.3B-T2V", "gpu": "T4", "max_seconds": 5}

    def _check_key(body):
        key = os.environ.get("VIDEO_KEY")
        if key and body.get("key") != key:
            return False
        return True

    @web.get("/health")
    def health():
        return {"ok": True, "model": "Wan2.1-1.3B-T2V", "gpu": "T4",
                "max_seconds": 5, "modes": ["text-to-video"]}

    @web.post("/generate")
    async def generate(body: dict):
        if not _check_key(body):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        job_id = uuid.uuid4().hex[:12]
        jobs[job_id] = {"status": "queued"}
        seed = body.get("seed", -1)
        generate_clip.spawn(
            job_id,
            body.get("prompt", ""),
            body.get("negative_prompt", ""),
            int(body.get("width") or 832),
            int(body.get("height") or 480),
            float(body.get("duration_seconds") or 5),
            int(body.get("steps") or 4),
            float(body.get("guidance") or 1.0),
            -1 if seed is None else int(seed),
            body.get("image_b64"),
        )
        return {"job_id": job_id}

    @web.get("/status/{job_id}")
    def status(job_id: str):
        st = jobs.get(job_id)
        if not st:
            return JSONResponse({"error": "unknown job"}, status_code=404)
        out = {"status": st["status"]}
        if st["status"] == "done":
            out["video_url"] = f"/download/{job_id}"
        if st.get("error"):
            out["error"] = st["error"]
        return out

    @web.get("/download/{job_id}")
    def download(job_id: str):
        path = f"/out/{job_id}.mp4"
        if not os.path.exists(path):
            return JSONResponse({"error": "not ready"}, status_code=404)
        return FileResponse(path, media_type="video/mp4",
                            filename=f"{job_id}.mp4")

    return web
