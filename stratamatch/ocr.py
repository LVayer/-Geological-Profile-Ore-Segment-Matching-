"""Local Windows OCR adapter. Suggestions never become verified lithology labels."""
import json
import os
import subprocess
import uuid
from PIL import Image
from .io import local

def recognize_text(image,page=0,region=None):
    if os.name!='nt':return {'status':'unavailable','reason':'Windows OCR is not available on this platform','lines':[]}
    folder=local('.tmp/ocr');folder.mkdir(parents=True,exist_ok=True)
    path=folder/(uuid.uuid4().hex+'.png')
    with Image.open(local(image)) as im:
        im.seek(page);im=im.convert('RGB')
        ox,oy=0,0
        if region:
            ox,oy,w,h=region
            if min(ox,oy)<0 or min(w,h)<1 or ox+w>im.width or oy+h>im.height:raise ValueError('Invalid OCR crop')
            im=im.crop((ox,oy,ox+w,oy+h))
        # Upscale tiny exported labels and respect the Windows OCR bitmap bound.
        factor=min(3.,2200/max(im.size))
        im.resize((max(1,round(im.width*factor)),max(1,round(im.height*factor))),Image.Resampling.LANCZOS).save(path)
    try:
        completed=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(local('tools/windows_ocr.ps1')),'-ImagePath',str(path)],
            capture_output=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW,encoding='utf-8',errors='replace')
        data=json.loads(completed.stdout.strip().lstrip('\ufeff'))
        for line in data.get('lines',[]):
            for word in line.get('words',[]):
                x,y,w,h=[float(v)/factor for v in word['box']];word['box']=[x+ox,y+oy,w,h]
        return data
    except (OSError,ValueError,subprocess.TimeoutExpired) as exc:
        return {'status':'unavailable','reason':type(exc).__name__,'lines':[]}
    finally:
        # The random temporary file is already resolved and confined by local().
        path.unlink(missing_ok=True)
