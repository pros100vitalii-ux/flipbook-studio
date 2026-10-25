import os, uuid, subprocess, shutil
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT=Path(__file__).resolve().parent
WORK=ROOT/"work"
WORK.mkdir(exist_ok=True)

# Accept either the correct path static/index.html or the accidental nested
# path static/static/index.html created during mobile GitHub editing.
STATIC_DIR=ROOT/"static"
if not (STATIC_DIR/"index.html").is_file() and (STATIC_DIR/"static"/"index.html").is_file():
    STATIC_DIR=STATIC_DIR/"static"

app=FastAPI(title="Real Flipbook Prototype")
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    index=STATIC_DIR/"index.html"
    if not index.is_file():
        raise HTTPException(status_code=500, detail=f"Не найден файл интерфейса: {index}. Проверьте, что static/index.html загружен в GitHub.")
    return index.read_text(encoding="utf-8")

def run(cmd):
    p=subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-2500:])
    return p

@app.post("/render")
async def render(source: UploadFile=File(...), template: UploadFile=File(...),
                 corners: str=Form("0.20,0.18,0.80,0.18,0.80,0.82,0.20,0.82"),
                 style: str=Form("pencil")):
    job=uuid.uuid4().hex
    d=WORK/job
    d.mkdir()
    src=d/"source"
    tpl=d/"template"
    src.write_bytes(await source.read())
    tpl.write_bytes(await template.read())
    try:
        # Probe source duration and cap at 30 sec.
        probe=subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(src)],capture_output=True,text=True)
        if probe.returncode: raise RuntimeError("Не удалось прочитать исходное видео.")
        duration=min(float(probe.stdout.strip()),30.0)
        cap=[float(x) for x in corners.split(",")]
        if len(cap)!=8 or any(x<0 or x>1 for x in cap):
            raise RuntimeError("Нужно указать 4 угла страницы в формате 8 чисел от 0 до 1.")
        # Template rendered to 1080p/30fps, duration limited to source duration.
        template_mp4=d/"template.mp4"
        run(["ffmpeg","-y","-i",str(tpl),"-t",str(duration),"-vf","scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,fps=30","-an","-c:v","libx264","-pix_fmt","yuv420p",str(template_mp4)])
        src_mp4=d/"source.mp4"
        run(["ffmpeg","-y","-i",str(src),"-t",str(duration),"-vf","scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,fps=15","-an","-c:v","libx264","-pix_fmt","yuv420p",str(src_mp4)])
        # Extract images. Prototype uses fixed page corners; tracking/page occlusion is a later milestone.
        frames_dir=d/"srcframes"; frames_dir.mkdir()
        run(["ffmpeg","-y","-i",str(src_mp4),str(frames_dir/"%06d.png")])
        cap_tpl=cv2.VideoCapture(str(template_mp4))
        fps=30
        out=d/"composite.mp4"
        writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*"mp4v"),fps,(1920,1080))
        frame_files=sorted(frames_dir.glob("*.png"))
        if not frame_files: raise RuntimeError("В исходном видео не найдены кадры.")
        src_imgs=[]
        for f in frame_files:
            im=cv2.imread(str(f))
            if im is None: continue
            if style=="pencil":
                gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
                inv=255-gray
                blur=cv2.GaussianBlur(inv,(21,21),0)
                sketch=cv2.divide(gray,255-blur,scale=256)
                im=cv2.cvtColor(sketch,cv2.COLOR_GRAY2BGR)
            src_imgs.append(im)
        if not src_imgs: raise RuntimeError("Не удалось обработать кадры исходного видео.")
        hsrc,wsrc=src_imgs[0].shape[:2]
        pts=np.array([[cap[0]*1920,cap[1]*1080],[cap[2]*1920,cap[3]*1080],[cap[4]*1920,cap[5]*1080],[cap[6]*1920,cap[7]*1080]],dtype=np.float32)
        target=np.array([[0,0],[639,0],[639,359],[0,359]],dtype=np.float32)
        M=cv2.getPerspectiveTransform(target,pts)
        i=0
        while True:
            ok,bg=cap_tpl.read()
            if not ok: break
            si=min(int(i*15/fps),len(src_imgs)-1)
            warped=cv2.warpPerspective(src_imgs[si],M,(1920,1080))
            mask_src=np.ones((360,640),dtype=np.uint8)*255
            mask=cv2.warpPerspective(mask_src,M,(1920,1080))
            # Simple overlay. Realistic occlusion/curved page tracking is not implemented in this first prototype.
            bg=cv2.bitwise_and(bg,bg,mask=cv2.bitwise_not(mask))
            fg=cv2.bitwise_and(warped,warped,mask=mask)
            bg=cv2.add(bg,fg)
            writer.write(bg)
            i+=1
            if i>=int(duration*fps): break
        cap_tpl.release(); writer.release()
        # Re-encode widely compatible MP4.
        final=d/"flipbook_fullhd.mp4"
        run(["ffmpeg","-y","-i",str(out),"-c:v","libx264","-preset","veryfast","-crf","21","-pix_fmt","yuv420p","-movflags","+faststart","-an",str(final)])
        return FileResponse(final, media_type="video/mp4", filename="flipbook_fullhd.mp4")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=int(os.getenv("PORT","8000")))
