"""Make an offline, seekable replay from recorded camera frames of the first match."""
import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image
from run_hockey_matches import play_point


def replay_data(checkpoint, receipt):
    data = {}
    for condition in ('after_hockey', 'before_hockey'):
        points = []
        score = [0, 0]
        for original in receipt[condition]['matches'][0]['points']:
            point, frames = play_point(original['seed'], checkpoint, condition == 'after_hockey',
                                       opponent=original['opponent'], record=True)
            if point != original:
                raise ValueError(f"failed to reproduce point {point['seed']}")
            # A lossless sprite sheet keeps every fourth actual camera frame and the final one.
            height, width = frames[0].shape[:2]
            columns = 12
            sheet = Image.new('RGB', (width * columns, height * ((len(frames)+columns-1)//columns)))
            for i, frame in enumerate(frames):
                sheet.paste(Image.fromarray(frame), (width*(i%columns), height*(i//columns)))
            buffer = io.BytesIO()
            sheet.convert('P',palette=Image.Palette.ADAPTIVE,colors=32).save(buffer,format='PNG',optimize=True)
            old = list(score)
            if point['score']:
                score[0 if point['score'] == 1 else 1] += 1
            points.append(dict(n=len(frames), start=old, end=list(score), reward=point['score'],
                               touches=point['own_touches'], image=base64.b64encode(buffer.getvalue()).decode('ascii')))
        data[condition] = points
    return data


HTML = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Air hockey match replay</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#0d1821;color:#e2e9e6;font:16px system-ui,sans-serif}
main{max-width:820px;margin:4vh auto;padding:20px}h1{font-size:clamp(26px,4vw,42px);margin:0 0 10px;letter-spacing:-.04em}
p{color:#b5c7cd;line-height:1.5}select,button{font:inherit;background:#1f3541;color:#f1f4f2;border:1px solid #627e88;border-radius:5px;padding:7px 12px}
label{display:inline-flex;align-items:center;gap:8px}nav,footer{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:16px 0}
canvas{display:block;aspect-ratio:4/3;width:100%;max-height:58vh;object-fit:contain;background:#11202e;image-rendering:pixelated}
section{background:#11202e;border:1px solid #315060;border-radius:8px;overflow:hidden}header{display:flex;justify-content:space-between;padding:12px 18px;font-variant-numeric:tabular-nums;border-bottom:1px solid #315060}
#score{font-weight:700}#seek{flex:1;min-width:120px;accent-color:#76d7c7}#point-label{color:#bed3da}
</style>
<main><h1>Air hockey match replay</h1><p>The red mallet is the learner. The blue mallet is the moving guard. First to seven goals wins.</p>
<nav><label>Policy <select id="policy"><option value="after_hockey">After hockey practice</option><option value="before_hockey">Before hockey practice</option></select></label>
<label>Point <select id="point"></select></label></nav>
<section><header><span id="score"></span><span id="point-label"></span></header>
<canvas id="rink" width="128" height="96" aria-label="Recorded air hockey camera"></canvas></section>
<footer><button id="play" type="button">Play</button><input id="seek" type="range" min="0" value="0" aria-label="Seek within point"><span id="clock"></span></footer>
<p>Recorded RGB frames from the first scheduled neutral-faceoff match. Both policies start with the same Pong wall estimate. The after-practice policy also carries its learned hockey shot values.</p></main>
<script>
const runs=DATA;
const policy=document.getElementById('policy'), point=document.getElementById('point'), seek=document.getElementById('seek'), play=document.getElementById('play');
const ctx=document.getElementById('rink').getContext('2d');
let image, clip, timer=null, ready=false, index=0;
function stop(){if(timer!==null)clearInterval(timer);timer=null;play.textContent='Play'}
function draw(){
 if(!ready)return;
 ctx.drawImage(image,(index%12)*128,Math.floor(index/12)*96,128,96,0,0,128,96);
 let last=index===clip.n-1, score=last?clip.end:clip.start;
 document.getElementById('score').textContent='Learner '+score[0]+'  —  '+score[1]+' Guard';
 document.getElementById('clock').textContent=(index/15).toFixed(1)+' s';
 document.getElementById('point-label').textContent=last?(clip.reward===1?'Learner goal':clip.reward===-1?'Guard goal':'Time limit'):'In play';
 seek.value=index;
}
function loadClip(auto=false){
 stop();ready=false;clip=runs[policy.value][Number(point.value)];index=0;seek.max=clip.n-1;
 const next=new Image();image=next;next.onload=()=>{if(image!==next)return;ready=true;draw();if(auto)start()};next.src='data:image/png;base64,'+clip.image;
}
function start(){if(!ready)return;if(index===clip.n-1)index=0;play.textContent='Pause';timer=setInterval(()=>{
 if(index<clip.n-1){index++;draw();return}
 const k=Number(point.value)+1;if(k<runs[policy.value].length){point.value=k;loadClip(true)}else stop();
},1000/15)}
function fillPoints(){
 const keep=Number(point.value)||0;point.replaceChildren();
 runs[policy.value].forEach((c,i)=>point.add(new Option(String(i+1),String(i))));
 point.value=Math.min(keep,point.options.length-1);loadClip()
}
policy.onchange=fillPoints;point.onchange=()=>loadClip();play.onclick=()=>timer===null?start():stop();
seek.oninput=()=>{stop();index=Number(seek.value);draw()};fillPoints();
</script></html>"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    runs = replay_data(json.loads(args.checkpoint.read_bytes()), json.loads(args.receipt.read_bytes()))
    args.output.write_text(HTML.replace('const runs=DATA;', 'const runs='+json.dumps(runs,separators=(',',':'))+';'))
