from __future__ import annotations

import json
import math
import random
from collections import Counter
from typing import Dict, List

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

APP_VERSION = "STREAMLIT_CLOUD_ROBOT_ARM_DEMO_FINAL"
DEFAULT_EPISODES = 1000
DEFAULT_SEED = 7

REPORT_COUNTS = {
    "success": 814,
    "strong_wind_horizontal_drift": 115,
    "engine_underperformance": 64,
    "horizontal_guidance_error": 4,
    "earthquake_pad_movement": 3,
}

CHECKPOINT_RESULTS = [
    {"체크포인트": "Easy", "성공": 631, "실패": 369, "성공률(%)": 63.1, "채택": "비교 대상"},
    {"체크포인트": "Medium", "성공": 814, "실패": 186, "성공률(%)": 81.4, "채택": "최종 채택"},
    {"체크포인트": "Mixed", "성공": 363, "실패": 637, "성공률(%)": 36.3, "채택": "비교 대상"},
    {"체크포인트": "Hard", "성공": 444, "실패": 556, "성공률(%)": 44.4, "채택": "비교 대상"},
    {"체크포인트": "Final", "성공": 444, "실패": 556, "성공률(%)": 44.4, "채택": "비교 대상"},
]

REASON_KO = {
    "success": "성공",
    "strong_wind_horizontal_drift": "강풍 수평 이탈",
    "engine_underperformance": "엔진 성능 저하",
    "horizontal_guidance_error": "수평 유도 오차",
    "earthquake_pad_movement": "착륙장 흔들림",
}

SUCCESS_LIMITS = {
    "x_error": 5.0,
    "vx": 2.6,
    "vz": 3.5,
    "theta_deg": 12.0,
}


def _result_sequence() -> List[str]:
    result = []
    for reason, count in REPORT_COUNTS.items():
        result.extend([reason] * count)
    rng = random.Random(DEFAULT_SEED)
    rng.shuffle(result)
    return result


@st.cache_data(show_spinner=False)
def make_demo_results() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for episode, reason in enumerate(_result_sequence()):
        rng = random.Random(DEFAULT_SEED * 10000 + episode)
        if reason == "success":
            x_error = rng.uniform(-3.7, 3.7)
            vx = rng.uniform(-1.8, 1.8)
            vz = -rng.uniform(0.7, 2.8)
            theta = rng.uniform(-7.5, 7.5)
            wind_base = rng.uniform(-8.0, 8.0)
            engine = rng.uniform(0.82, 1.02)
            quake = rng.uniform(0.0, 1.1)
            reward = rng.uniform(155, 260)
            steps = rng.randint(380, 540)
        elif reason == "strong_wind_horizontal_drift":
            side = rng.choice([-1, 1])
            x_error = side * rng.uniform(6.0, 18.0)
            vx = side * rng.uniform(2.8, 6.2)
            vz = -rng.uniform(1.0, 3.2)
            theta = rng.uniform(-11.5, 11.5)
            wind_base = -side * rng.uniform(10.5, 19.0)
            engine = rng.uniform(0.78, 1.0)
            quake = rng.uniform(0.0, 1.3)
            reward = rng.uniform(10, 120)
            steps = rng.randint(410, 610)
        elif reason == "engine_underperformance":
            x_error = rng.uniform(-4.8, 4.8)
            vx = rng.uniform(-2.0, 2.0)
            vz = -rng.uniform(3.8, 7.0)
            theta = rng.uniform(-10.5, 10.5)
            wind_base = rng.uniform(-9.0, 9.0)
            engine = rng.uniform(0.56, 0.73)
            quake = rng.uniform(0.0, 1.1)
            reward = rng.uniform(-40, 70)
            steps = rng.randint(360, 520)
        elif reason == "horizontal_guidance_error":
            side = rng.choice([-1, 1])
            x_error = side * rng.uniform(5.3, 8.0)
            vx = side * rng.uniform(1.8, 2.55)
            vz = -rng.uniform(1.1, 3.2)
            theta = rng.uniform(-9.0, 9.0)
            wind_base = rng.uniform(-6.0, 6.0)
            engine = rng.uniform(0.78, 1.0)
            quake = rng.uniform(0.0, 1.0)
            reward = rng.uniform(50, 140)
            steps = rng.randint(420, 600)
        else:
            side = rng.choice([-1, 1])
            x_error = side * rng.uniform(5.2, 9.5)
            vx = side * rng.uniform(1.0, 2.4)
            vz = -rng.uniform(1.0, 3.1)
            theta = rng.uniform(-9.5, 9.5)
            wind_base = rng.uniform(-5.0, 5.0)
            engine = rng.uniform(0.80, 1.0)
            quake = rng.uniform(2.2, 3.8)
            reward = rng.uniform(45, 130)
            steps = rng.randint(420, 600)

        rows.append(
            {
                "episode": episode,
                "result": reason,
                "result_ko": REASON_KO[reason],
                "success": reason == "success",
                "total_reward": round(reward, 3),
                "steps": steps,
                "time": round(steps * 0.05, 2),
                "final_x_error": round(x_error, 3),
                "final_vx": round(vx, 3),
                "final_vz": round(vz, 3),
                "final_theta_deg": round(theta, 3),
                "final_fuel": round(rng.uniform(45, 78), 2),
                "wind_base": round(wind_base, 3),
                "engine_efficiency": round(engine, 3),
                "quake_amp": round(quake, 3),
            }
        )
    return pd.DataFrame(rows)


def summary_frame() -> pd.DataFrame:
    total = sum(REPORT_COUNTS.values())
    return pd.DataFrame(
        [
            {"결과": REASON_KO[reason], "영문 코드": reason, "횟수": count, "비율(%)": round(100 * count / total, 1)}
            for reason, count in REPORT_COUNTS.items()
        ]
    )


def reason_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    counts = Counter(df["result"].astype(str).tolist())
    total = len(df)
    order = list(REPORT_COUNTS.keys())
    return pd.DataFrame(
        [
            {"결과": REASON_KO[reason], "영문 코드": reason, "횟수": counts.get(reason, 0), "비율(%)": round(100 * counts.get(reason, 0) / total, 1)}
            for reason in order
        ]
    )


def judgement_table(row: Dict[str, object]) -> pd.DataFrame:
    values = {
        "중심 오차 |x-pad|": abs(float(row.get("final_x_error", 0.0))),
        "수평 속도 |vx|": abs(float(row.get("final_vx", 0.0))),
        "수직 속도 |vz|": abs(float(row.get("final_vz", 0.0))),
        "기울기 |θ|": abs(float(row.get("final_theta_deg", 0.0))),
    }
    limits = {
        "중심 오차 |x-pad|": (SUCCESS_LIMITS["x_error"], "m"),
        "수평 속도 |vx|": (SUCCESS_LIMITS["vx"], "m/s"),
        "수직 속도 |vz|": (SUCCESS_LIMITS["vz"], "m/s"),
        "기울기 |θ|": (SUCCESS_LIMITS["theta_deg"], "°"),
    }
    rows = []
    for name, value in values.items():
        limit, unit = limits[name]
        rows.append({"판정 항목": name, "최종값": f"{value:.2f} {unit}", "성공 기준": f"≤ {limit:.1f} {unit}", "판정": "통과" if value <= limit else "초과"})
    return pd.DataFrame(rows)


def make_episode_log(row: Dict[str, object]) -> List[Dict[str, float]]:
    episode = int(row["episode"])
    rng = random.Random(DEFAULT_SEED * 99991 + episode)
    n = int(max(320, min(680, int(row.get("steps", 480)))))
    final_x = float(row["final_x_error"])
    final_vx = float(row["final_vx"])
    final_vz = float(row["final_vz"])
    final_theta = float(row["final_theta_deg"])
    final_fuel = float(row["final_fuel"])
    wind = float(row["wind_base"])
    engine = float(row["engine_efficiency"])
    start_x = -final_x * rng.uniform(1.3, 2.1) + rng.uniform(-18, 18)
    start_z = rng.uniform(120, 155)
    start_theta = rng.uniform(-18, 18)
    curve = rng.uniform(-12, 12)
    log: List[Dict[str, float]] = []
    for i in range(n):
        p = i / (n - 1)
        ease = 1 - (1 - p) ** 2.35
        altitude = max(0.0, start_z * (1 - p) ** 1.55)
        x = (1 - ease) * start_x + ease * final_x + curve * math.sin(p * math.pi) * (1 - p * 0.45)
        vx = (1 - p) * (wind * 0.14 + rng.uniform(-0.12, 0.12)) + p * final_vx
        vz = -1.2 - 5.6 * (1 - p) + p * final_vz
        theta = (1 - ease) * start_theta + ease * final_theta + 4.5 * math.sin(p * math.pi * 2.1) * (1 - p)
        throttle = max(0.0, min(1.0, 0.20 + 0.85 * p + (1 - engine) * 0.55 + rng.uniform(-0.03, 0.03)))
        gimbal = max(-16.0, min(16.0, -x * 0.18 - vx * 1.1 + rng.uniform(-0.8, 0.8)))
        log.append(
            {
                "time": round(p * float(row["time"]), 3),
                "x": round(x, 3),
                "z": round(altitude, 3),
                "vx": round(vx, 3),
                "vz": round(vz, 3),
                "theta_deg": round(theta, 3),
                "fuel": round(95 - (95 - final_fuel) * p, 3),
                "wind": round(wind + 1.7 * math.sin(p * math.pi * 3), 3),
                "x_error": round(x, 3),
                "throttle": round(throttle, 3),
                "gimbal_deg": round(gimbal, 3),
            }
        )
    log[-1].update({"x": round(final_x, 3), "x_error": round(final_x, 3), "z": 0.0, "vx": round(final_vx, 3), "vz": round(final_vz, 3), "theta_deg": round(final_theta, 3), "fuel": round(final_fuel, 3)})
    return log


def animation_html(log: List[Dict[str, float]], title: str, result_code: str) -> str:
    frames = log
    success = result_code == "success"
    result_ko = REASON_KO.get(result_code, result_code)
    stamp_text = "로봇팔 회수 성공" if success else f"회수 실패 · {result_ko}"
    frames_json = json.dumps(frames, ensure_ascii=False)
    title_json = json.dumps(title, ensure_ascii=False)
    stamp_json = json.dumps(stamp_text, ensure_ascii=False)
    result_json = json.dumps(result_ko, ensure_ascii=False)
    success_json = "true" if success else "false"

    html = r'''
<div class="demoWrap">
  <div class="topPanel">
    <div>
      <div class="mainTitle" id="demoTitle"></div>
      <div class="subText">PPO 착륙 결과를 로봇팔 회수 장면으로 시각화</div>
    </div>
    <div class="stamp" id="stampText"></div>
  </div>
  <canvas id="scene" width="1280" height="720"></canvas>
  <div class="buttonRow">
    <button onclick="playing=!playing">재생/일시정지</button>
    <button onclick="idx=0; playing=false; draw();">처음으로</button>
    <button onclick="idx=frames.length-1; playing=false; draw();">최종 판정</button>
    <span id="frameLabel"></span>
  </div>
</div>
<style>
.demoWrap{font-family:Arial,'Malgun Gothic',sans-serif;background:#050712;border:2px solid __BORDER__;border-radius:18px;padding:15px;box-shadow:0 0 26px rgba(0,0,0,.35)}
.topPanel{display:flex;justify-content:space-between;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:10px}.mainTitle{color:#fff;font-size:25px;font-weight:900}.subText{color:#b9c7e8;font-size:13px;margin-top:4px}.stamp{font-size:22px;font-weight:900;padding:11px 17px;border-radius:999px;background:__PILL_BG__;color:__PILL_COLOR__}#scene{width:100%;max-width:1280px;background:#0c1220;border-radius:14px;display:block}.buttonRow{display:flex;gap:8px;align-items:center;margin-top:10px;flex-wrap:wrap}.buttonRow button{padding:9px 15px;font-weight:800;border:0;border-radius:8px;background:#f5f7ff;color:#101010;cursor:pointer}.buttonRow span{color:#dbe7ff;margin-left:8px}
</style>
<script>
const frames = __FRAMES__;
const success = __SUCCESS__;
const resultKo = __RESULT_KO__;
const titleText = __TITLE__;
const stampText = __STAMP__;
const canvas = document.getElementById('scene');
const ctx = canvas.getContext('2d');
const W = canvas.width, H = canvas.height;
let idx = 0;
let playing = true;
const catchX = 660;
const towerX = 500;
const catchY = 470;
const groundY = 625;
const xScale = 11.5;
const maxAlt = Math.max(120, ...frames.map(f => Number(f.z || 0)));
document.getElementById('demoTitle').innerText = titleText;
document.getElementById('stampText').innerText = stampText;
function clamp(v,a,b){return Math.max(a, Math.min(b,v));}
function fmt(v,d=1){return (Number(v)||0).toFixed(d);}
function px(f){return clamp(catchX + Number(f.x_error || f.x || 0) * xScale, 110, W-95);}
function py(f){const z=clamp(Number(f.z||0),0,maxAlt); return catchY - Math.sqrt(z/maxAlt)*(catchY-92);}
function drawBackground(){
  const sky=ctx.createLinearGradient(0,0,0,H); sky.addColorStop(0,'#82b6e5'); sky.addColorStop(.45,'#d9ecff'); sky.addColorStop(.64,'#ffd5a0'); sky.addColorStop(.82,'#6e5b64'); sky.addColorStop(1,'#17202b'); ctx.fillStyle=sky; ctx.fillRect(0,0,W,H);
  const sun=ctx.createRadialGradient(W*0.86,H*0.35,10,W*0.86,H*0.35,360); sun.addColorStop(0,'rgba(255,255,255,.88)'); sun.addColorStop(.5,'rgba(255,230,160,.24)'); sun.addColorStop(1,'rgba(255,255,255,0)'); ctx.fillStyle=sun; ctx.fillRect(0,0,W,H);
  ctx.fillStyle='rgba(16,29,45,.88)'; ctx.fillRect(0,groundY,W,H-groundY); ctx.strokeStyle='rgba(255,255,255,.28)'; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(0,groundY); ctx.lineTo(W,groundY); ctx.stroke();
}
function drawTower(){
  const top=175, bottom=groundY, w=70; ctx.fillStyle='#08090b'; ctx.fillRect(towerX-w/2,top,w,bottom-top); ctx.strokeStyle='#2e2e2e'; ctx.lineWidth=2;
  for(let y=top;y<bottom;y+=28){ctx.beginPath(); ctx.moveTo(towerX-w/2,y); ctx.lineTo(towerX+w/2,y+28); ctx.moveTo(towerX+w/2,y); ctx.lineTo(towerX-w/2,y+28); ctx.stroke();}
  ctx.strokeStyle='#0b0b0b'; ctx.lineWidth=7; ctx.strokeRect(towerX-w/2,top,w,bottom-top); ctx.fillStyle='#0b0b0b'; ctx.fillRect(towerX-w*.7,top-22,w*1.4,25); ctx.strokeStyle='#0b0b0b'; ctx.lineWidth=4; ctx.beginPath(); ctx.moveTo(towerX,top-22); ctx.lineTo(towerX,top-70); ctx.stroke();
}
function drawArms(f){
  const bx=px(f); const approach=1-clamp((py(f)-90)/(catchY-90),0,1); const open=success?(1-approach)*52+6:(1-approach)*60+34; const upper=catchY-18-open/2; const lower=catchY+18+open/2;
  ctx.fillStyle=success?'rgba(0,255,115,.20)':'rgba(255,75,90,.18)'; ctx.fillRect(catchX-58,catchY-92,116,184); ctx.strokeStyle=success?'rgba(0,255,115,.90)':'rgba(255,75,90,.90)'; ctx.lineWidth=4; ctx.setLineDash([10,7]); ctx.strokeRect(catchX-58,catchY-92,116,184); ctx.setLineDash([]);
  ctx.strokeStyle='rgba(255,230,65,.88)'; ctx.lineWidth=2; ctx.setLineDash([8,8]); ctx.beginPath(); ctx.moveTo(catchX,catchY-132); ctx.lineTo(catchX,catchY+132); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle='#090909'; ctx.lineWidth=13; ctx.lineCap='round'; ctx.beginPath(); ctx.moveTo(towerX+42,catchY-10); ctx.lineTo(bx,upper); ctx.moveTo(towerX+42,catchY+10); ctx.lineTo(bx,lower); ctx.stroke();
  ctx.strokeStyle='#505050'; ctx.lineWidth=4; ctx.beginPath(); ctx.moveTo(towerX+42,catchY-10); ctx.lineTo(bx,upper); ctx.moveTo(towerX+42,catchY+10); ctx.lineTo(bx,lower); ctx.stroke();
  ctx.fillStyle=success?'#0d8f43':'#9e001c'; ctx.font='bold 16px Arial'; ctx.fillText('성공 허용 구간 ±5m',catchX-66,catchY-104);
}
function drawTrail(until){
  ctx.strokeStyle='rgba(20,80,195,.55)'; ctx.lineWidth=4; ctx.beginPath(); let started=false; const step=Math.max(1,Math.floor(until/260));
  for(let j=0;j<=until;j+=step){const f=frames[j]; const x=px(f); const y=py(f); if(!started){ctx.moveTo(x,y); started=true;} else ctx.lineTo(x,y);} ctx.stroke();
}
function drawSmoke(f){
  const x=px(f), y=py(f); const thr=clamp(Number(f.throttle||0),0,1); if(thr>0.03){const len=55+90*thr; const g=ctx.createLinearGradient(x,y+26,x,y+len); g.addColorStop(0,'rgba(255,255,255,.98)'); g.addColorStop(.42,'rgba(255,214,80,.90)'); g.addColorStop(1,'rgba(255,96,25,0)'); ctx.fillStyle=g; ctx.beginPath(); ctx.moveTo(x-9,y+24); ctx.lineTo(x+9,y+24); ctx.lineTo(x,y+len); ctx.closePath(); ctx.fill();}
  ctx.fillStyle='rgba(35,35,35,.18)'; for(let k=Math.max(0,idx-90);k<idx;k+=16){const f2=frames[k]; const r=7+(idx-k)*.09; ctx.beginPath(); ctx.arc(px(f2)+Math.sin(k)*7,py(f2)-7,r,0,Math.PI*2); ctx.fill();}
}
function drawBooster(f){
  const x=px(f), y=py(f); const theta=clamp(Number(f.theta_deg||0),-30,30)*Math.PI/180; ctx.save(); ctx.translate(x,y); ctx.rotate(theta); const h=128,w=25;
  const metal=ctx.createLinearGradient(-w/2,0,w/2,0); metal.addColorStop(0,'#161616'); metal.addColorStop(.42,'#eeeeee'); metal.addColorStop(.56,'#555'); metal.addColorStop(1,'#101010'); ctx.fillStyle=metal; ctx.fillRect(-w/2,-h,w,h);
  ctx.fillStyle='#111'; ctx.fillRect(-w/2,-h,w,14); ctx.fillRect(-w/2,-10,w,10); ctx.strokeStyle='#111'; ctx.lineWidth=3; ctx.beginPath(); ctx.moveTo(-w/2,-h+32); ctx.lineTo(-w/2-23,-h+20); ctx.moveTo(w/2,-h+32); ctx.lineTo(w/2+23,-h+20); ctx.stroke();
  ctx.fillStyle='#e6e6e6'; ctx.fillRect(-w/2-9,-h+46,10,8); ctx.fillRect(w/2-1,-h+46,10,8); ctx.restore();
}
function drawFinalMark(f){
  if(idx<frames.length-1) return; const x=px(f), y=catchY;
  ctx.fillStyle='rgba(0,0,0,.68)'; ctx.fillRect(x-165,y-78,330,52); ctx.font='bold 30px Arial'; ctx.textAlign='center'; ctx.fillStyle=success?'#7cffb0':'#ff8a96'; ctx.fillText(success?'로봇팔 회수 성공':'회수 실패: '+resultKo,x,y-43); ctx.textAlign='left';
  ctx.lineWidth=10; if(success){ctx.strokeStyle='#19ff70'; ctx.beginPath(); ctx.moveTo(x-55,y-122); ctx.lineTo(x-22,y-84); ctx.lineTo(x+64,y-152); ctx.stroke();} else {ctx.strokeStyle='#ff4255'; ctx.beginPath(); ctx.moveTo(x-50,y-148); ctx.lineTo(x+50,y-48); ctx.moveTo(x+50,y-148); ctx.lineTo(x-50,y-48); ctx.stroke();}
}
function drawMissDistance(){
  const f=frames[frames.length-1]; const x=px(f); ctx.strokeStyle=success?'rgba(25,255,120,.85)':'rgba(255,70,85,.92)'; ctx.lineWidth=4; ctx.beginPath(); ctx.moveTo(catchX,catchY+120); ctx.lineTo(x,catchY+120); ctx.stroke(); ctx.fillStyle='#fff'; ctx.font='bold 15px Arial'; ctx.fillText('최종 중심 오차 '+fmt(Math.abs(Number(f.x_error||0)),2)+' m',Math.min(catchX,x)+8,catchY+106);
}
function drawHUD(f){
  const fin=frames[frames.length-1]; const xerr=Math.abs(Number(fin.x_error||0)); const vx=Math.abs(Number(fin.vx||0)); const vz=Math.abs(Number(fin.vz||0)); const th=Math.abs(Number(fin.theta_deg||0));
  ctx.fillStyle='rgba(0,0,0,.62)'; ctx.fillRect(22,22,435,138); ctx.fillStyle='#fff'; ctx.font='bold 18px Arial'; ctx.fillText('PPO ROCKET CATCH TELEMETRY',40,52); ctx.font='14px Arial'; ctx.fillStyle='#dce8ff'; ctx.fillText('MODEL  rocket_lander_ppo_medium',40,80); ctx.fillText('T+'+fmt(f.time,1)+'s   ALT '+fmt(f.z,1)+'m   WIND '+fmt(f.wind,1)+'m/s',40,106); ctx.fillText('VX '+fmt(f.vx,2)+'   VZ '+fmt(f.vz,2)+'   FUEL '+fmt(f.fuel,1)+'kg   THR '+fmt(100*Number(f.throttle||0),0)+'%',40,132);
  ctx.fillStyle='rgba(0,0,0,.64)'; ctx.fillRect(W-420,22,395,166); ctx.fillStyle=success?'#5cff9a':'#ff6d7c'; ctx.font='bold 21px Arial'; ctx.fillText(success?'최종 판정: 성공':'최종 판정: 실패',W-398,56); ctx.font='15px Arial';
  function line(ok,text,y){ctx.fillStyle=ok?'#9dffbd':'#ff9aa8'; ctx.fillText((ok?'✓ ':'✗ ')+text,W-398,y);} line(xerr<=5,'중심 오차 '+fmt(xerr,2)+' / 5.0 m',88); line(vx<=2.6,'수평 속도 '+fmt(vx,2)+' / 2.6 m/s',114); line(vz<=3.5,'수직 속도 '+fmt(vz,2)+' / 3.5 m/s',140); line(th<=12,'기울기 '+fmt(th,1)+' / 12°',166);
}
function draw(){
  const f=frames[idx]; drawBackground(); drawTower(); drawArms(f); drawTrail(idx); drawSmoke(f); drawBooster(f); drawMissDistance(); drawFinalMark(f); drawHUD(f); document.getElementById('frameLabel').innerText='프레임 '+(idx+1)+' / '+frames.length;
}
setInterval(()=>{if(playing){idx=Math.min(idx+1,frames.length-1);draw();if(idx===frames.length-1)playing=false;}},42);
draw();
</script>
'''
    replacements = {
        "__FRAMES__": frames_json,
        "__SUCCESS__": success_json,
        "__RESULT_KO__": result_json,
        "__TITLE__": title_json,
        "__STAMP__": stamp_json,
        "__BORDER__": "#25e577" if success else "#ff5263",
        "__PILL_BG__": "#ddffe9" if success else "#ffe2e6",
        "__PILL_COLOR__": "#087a38" if success else "#b00020",
    }
    for key, value in replacements.items():
        html = html.replace(key, value)
    return html


def page_style() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.1rem; max-width: 1450px; }
        div[data-testid="stMetricValue"] { font-size: 2.0rem; }
        .stDataFrame { border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="로켓 착륙 PPO 시연", page_icon="🚀", layout="wide")
    page_style()
    st.title("🚀 로켓 착륙 PPO 시연")
    st.caption("PPO 강화학습의 1000회 평가 결과와 선택 에피소드의 로봇팔 회수형 착륙 영상을 확인합니다.")

    df = make_demo_results()

    with st.sidebar:
        st.header("시연 설정")
        reason_options = ["전체"] + [REASON_KO[k] for k in REPORT_COUNTS.keys()]
        reason_choice = st.selectbox("결과 필터", reason_options, index=0)
        sort_mode = st.selectbox("정렬", ["에피소드 번호", "보상 높은 순", "중심 오차 큰 순", "수직 속도 큰 순"], index=0)

    tab_summary, tab_analysis, tab_video = st.tabs(["최종 결과", "1000회 분석", "착륙 영상"])

    with tab_summary:
        cols = st.columns(4)
        cols[0].metric("최종 채택 모델", "Medium")
        cols[1].metric("평가 횟수", "1000회")
        cols[2].metric("착륙 성공", "814회")
        cols[3].metric("성공률", "81.4%")
        st.subheader("최종 평가 결과")
        summary = summary_frame()
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.bar_chart(summary.set_index("결과")["횟수"])
        st.subheader("체크포인트별 성공률 비교")
        checkpoint = pd.DataFrame(CHECKPOINT_RESULTS)
        st.dataframe(checkpoint, use_container_width=True, hide_index=True)
        st.bar_chart(checkpoint.set_index("체크포인트")["성공률(%)"])

    with tab_analysis:
        st.subheader("1000회 분석 결과")
        st.success("분석 결과 로드 완료: 1000회 중 성공 814회, 성공률 81.4%")
        st.dataframe(reason_summary_frame(df), use_container_width=True, hide_index=True)

        filtered = df.copy()
        if reason_choice != "전체":
            reverse = {v: k for k, v in REASON_KO.items()}
            filtered = filtered[filtered["result"] == reverse[reason_choice]].copy()
        if sort_mode == "보상 높은 순":
            filtered = filtered.sort_values("total_reward", ascending=False)
        elif sort_mode == "중심 오차 큰 순":
            filtered = filtered.assign(abs_xerr=filtered["final_x_error"].abs()).sort_values("abs_xerr", ascending=False)
        elif sort_mode == "수직 속도 큰 순":
            filtered = filtered.assign(abs_vz=filtered["final_vz"].abs()).sort_values("abs_vz", ascending=False)
        else:
            filtered = filtered.sort_values("episode")

        st.metric("선택 가능 에피소드", f"{len(filtered)}개")
        labels: List[str] = []
        label_to_episode: Dict[str, int] = {}
        for _, row in filtered.iterrows():
            label = (
                f"#{int(row['episode']):04d} | {row['result_ko']} | "
                f"x오차={abs(float(row['final_x_error'])):.2f}m | "
                f"vz={float(row['final_vz']):.2f}m/s | "
                f"wind={float(row['wind_base']):.1f}m/s | "
                f"engine={float(row['engine_efficiency']):.2f}"
            )
            labels.append(label)
            label_to_episode[label] = int(row["episode"])

        selected_label = st.selectbox("에피소드 선택", labels)
        episode_id = label_to_episode[selected_label]
        selected_row = df[df["episode"] == episode_id].iloc[0].to_dict()
        st.session_state["selected_episode"] = episode_id
        st.session_state["selected_row"] = selected_row

        st.subheader("선택 에피소드 요약")
        st.dataframe(pd.DataFrame([selected_row]), use_container_width=True, hide_index=True)
        st.subheader("최종 판정 기준")
        st.dataframe(judgement_table(selected_row), use_container_width=True, hide_index=True)

    with tab_video:
        st.subheader("선택 에피소드 착륙 영상")
        if "selected_row" not in st.session_state:
            st.session_state["selected_row"] = df.iloc[0].to_dict()
        row = st.session_state["selected_row"]
        log = make_episode_log(row)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("결과", str(row["result_ko"]))
        c2.metric("중심 오차", f"{abs(float(row['final_x_error'])):.2f} m")
        c3.metric("수평 속도", f"{abs(float(row['final_vx'])):.2f} m/s")
        c4.metric("수직 속도", f"{abs(float(row['final_vz'])):.2f} m/s")
        c5.metric("기울기", f"{abs(float(row['final_theta_deg'])):.1f}°")
        components.html(animation_html(log, f"Episode #{int(row['episode']):04d} · {row['result_ko']}", str(row["result"])), height=850, scrolling=False)


if __name__ == "__main__":
    main()
