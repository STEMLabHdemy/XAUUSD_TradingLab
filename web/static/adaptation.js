const fmt=n=>Number(n||0).toLocaleString('it-IT',{minimumFractionDigits:2,maximumFractionDigits:2});
let payload,pc,ec,ps,es,bs,marks;
function chart(el){return LightweightCharts.createChart(el,{width:el.clientWidth,height:el.clientHeight,layout:{background:{color:'#0b1220'},textColor:'#aebed4'},grid:{vertLines:{color:'#1f3049'},horzLines:{color:'#1f3049'}},timeScale:{timeVisible:true},rightPriceScale:{borderColor:'#334766'}})}
function windowSize(){let value=document.querySelector('#chart-window').value;return value==='all'?0:Number(value)}
const epoch=value=>Math.floor(Date.parse(value)/1000);
// Rendering 90k M1 bars in a narrow canvas makes every candle sub-pixel. We
// aggregate only the visual series; the replay and every trade remain intact.
function visualCandles(source){
  const maxBars=2500,step=Math.max(1,Math.ceil(source.length/maxBars)),bars=[],timeMap=new Map();
  for(let i=0;i<source.length;i+=step){
    const rows=source.slice(i,i+step),first=rows[0],last=rows[rows.length-1];
    const bar={time:first.time,open:first.open,high:Math.max(...rows.map(x=>x.high)),low:Math.min(...rows.map(x=>x.low)),close:last.close};
    bars.push(bar);rows.forEach(row=>timeMap.set(row.time,bar.time));
  }
  return {bars,timeMap};
}
function visualLine(source,cut){
  const points=source.filter(x=>epoch(x.datetime_utc)>=cut),step=Math.max(1,Math.ceil(points.length/9000)),result=[];
  for(let i=0;i<points.length;i+=step){const last=points[Math.min(i+step-1,points.length-1)];result.push({time:epoch(last.datetime_utc),value:last.equity})}
  return result;
}
function render(){
  const clean=payload.candles.filter(bar=>[bar.open,bar.high,bar.low,bar.close].every(Number.isFinite) && bar.low>0 && bar.high>=bar.low);
  let n=windowSize(),source=n?clean.slice(-n):clean;
  if(!source.length)throw Error('Nessuna candela valida nel periodo selezionato');
  let cut=source[0].time,visual=visualCandles(source),mapTime=time=>visual.timeMap.get(time)||visual.bars[0].time;
  ps.setData(visual.bars);
  const rawMarkers=payload.adaptive.trades.filter(t=>epoch(t.exit_time)>=cut).flatMap(t=>[
    {time:mapTime(epoch(t.entry_time)),position:t.side==='LONG'?'belowBar':'aboveBar',color:t.side==='LONG'?'#22c55e':'#ef4444',shape:t.side==='LONG'?'arrowUp':'arrowDown',text:t.side==='LONG'?'BUY':'SELL'},
    {time:mapTime(epoch(t.exit_time)),position:'aboveBar',color:'#f59e0b',shape:'circle',text:'EXIT'}
  ]).sort((a,b)=>a.time-b.time);
  // A visual bar can contain more than one trade after aggregation. The chart
  // marker API requires unique times, so retain one explicit marker and show
  // the number of additional events in that same visual bar.
  const grouped=new Map();
  rawMarkers.forEach(marker=>{
    let group=grouped.get(marker.time);
    if(!group){group={...marker,entries:[],exits:0};grouped.set(marker.time,group)}
    if(marker.text==='EXIT')group.exits++;else group.entries.push(marker.text);
  });
  const markerRows=[...grouped.values()].map(marker=>{
    const entryCount=marker.entries.length, entryLabel=entryCount ? `${marker.entries[0]}${entryCount>1?` ×${entryCount}`:''}` : '';
    const exitLabel=marker.exits ? `EXIT${marker.exits>1?` ×${marker.exits}`:''}` : '';
    const text=[entryLabel,exitLabel].filter(Boolean).join(' · ');
    return {...marker,text,position:entryCount?'belowBar':'aboveBar',shape:entryCount?(marker.entries[0]==='BUY'?'arrowUp':'arrowDown'):'circle',color:entryCount?(marker.entries[0]==='BUY'?'#22c55e':'#ef4444'):'#f59e0b'};
  });
  marks.setMarkers(markerRows);
  es.setData(visualLine(payload.adaptive.equity,cut));bs.setData(visualLine(payload.baseline.equity,cut));
  requestAnimationFrame(()=>{pc.timeScale().fitContent();ec.timeScale().fitContent()});
}
async function load(){try{let q=new URLSearchParams(location.search),id=q.get('candidate'),segment=q.get('segment')||'in_sample',url=id?'/api/research/adaptive-details/'+encodeURIComponent(id)+'?segment='+segment:'/api/research/adaptation';let r=await fetch(url),d=await r.json();if(!r.ok)throw Error(d.detail);payload=d;let a=d.adaptive.metrics,b=d.baseline.metrics;document.querySelector('#status').textContent=`${id||'Replay base'} · ${segment==='out_of_sample'?'fuori campione':'in sample'}`;document.querySelector('#status').style.color='#34d399';document.querySelector('#segment-title').textContent=`${id||'Replay base'} · ${segment==='out_of_sample'?'fuori campione':'in sample'}`;let basePath=id?'/adaptation?candidate='+encodeURIComponent(id):'/adaptation';document.querySelector('#in-sample').href=basePath;document.querySelector('#out-of-sample').href=basePath+(id?'&':'?')+'segment=out_of_sample';document.querySelector('#metrics').innerHTML=[['Base PnL',`${b.net_pnl>=0?'+':''}${fmt(b.net_pnl)} USD`],['Adaptive PnL',`${a.net_pnl>=0?'+':''}${fmt(a.net_pnl)} USD`],['PF base → adaptive',`${fmt(b.profit_factor)} → ${fmt(a.profit_factor)}`],['DD base → adaptive',`${(b.max_drawdown*100).toFixed(2)}% → ${(a.max_drawdown*100).toFixed(2)}%`],['Trade base → adaptive',`${b.trades} → ${a.trades}`],['Eventi filtrati',`${d.coverage.base_events} → ${d.coverage.gated_events}`]].map(x=>`<article><span>${x[0]}</span><b>${x[1]}</b></article>`).join('');pc=chart(document.querySelector('#price-chart'));ps=pc.addSeries(LightweightCharts.CandlestickSeries,{upColor:'#34d399',downColor:'#fb7185',borderVisible:false,wickUpColor:'#34d399',wickDownColor:'#fb7185'});marks=LightweightCharts.createSeriesMarkers(ps,[]);ec=chart(document.querySelector('#equity-chart'));es=ec.addSeries(LightweightCharts.LineSeries,{color:'#34d399',lineWidth:2,title:'Adaptive'});bs=ec.addSeries(LightweightCharts.LineSeries,{color:'#64748b',lineWidth:1,title:'Base'});document.querySelector('#chart-window').onchange=render;render()}catch(e){document.querySelector('#status').textContent='Errore: '+e.message;document.querySelector('#status').style.color='#fb7185'}}load();
