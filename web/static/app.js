const host=document.querySelector('#chart');
const roma=t=>new Date((typeof t==='number'?t:t.timestamp)*1000).toLocaleTimeString('it-IT',{timeZone:'Europe/Rome',hour:'2-digit',minute:'2-digit'});
const chart=LightweightCharts.createChart(host,{width:host.clientWidth,height:host.clientHeight,layout:{background:{color:'#0b1220'},textColor:'#aebed4'},grid:{vertLines:{color:'#1f3049'},horzLines:{color:'#1f3049'}},rightPriceScale:{borderColor:'#334766'},localization:{timeFormatter:roma},timeScale:{borderColor:'#334766',timeVisible:true,secondsVisible:false,tickMarkFormatter:roma},crosshair:{mode:LightweightCharts.CrosshairMode.Normal}});
const series=chart.addSeries(LightweightCharts.CandlestickSeries,{upColor:'#34d399',downColor:'#fb7185',borderVisible:false,wickUpColor:'#34d399',wickDownColor:'#fb7185'});
const volume=chart.addSeries(LightweightCharts.HistogramSeries,{priceFormat:{type:'volume'},priceScaleId:'vol'});volume.priceScale().applyOptions({scaleMargins:{top:.8,bottom:0}});
const ema20=chart.addSeries(LightweightCharts.LineSeries,{color:'#fb923c',lineWidth:2,lastValueVisible:true,priceLineVisible:false,title:'EMA 20'});
const ema100=chart.addSeries(LightweightCharts.LineSeries,{color:'#4ade80',lineWidth:2,lastValueVisible:true,priceLineVisible:false,title:'EMA 100'});
const markerPlugin=LightweightCharts.createSeriesMarkers(series,[]);let lines=[],chosen='',chartWindow='50',first=true;
const fmt=n=>Number(n||0).toLocaleString('it-IT',{minimumFractionDigits:2,maximumFractionDigits:2});
function levels(items){lines.forEach(line=>series.removePriceLine(line));lines=items.map(item=>series.createPriceLine({price:item.price,color:item.color,lineWidth:1,lineStyle:item.line_style,axisLabelVisible:true,title:item.title}))}
function probability(id,value){const el=document.querySelector(id);if(el)el.textContent=value==null?'-':(value*100).toFixed(1)+'%'}
function paint(d){
 const status=document.querySelector('#status'),health=d.paper_status||{},age=health.updated_at_utc?Date.now()-new Date(health.updated_at_utc).getTime():Infinity,healthy=age<90000&&String(health.message||'').includes('attivo'),marketStale=Boolean(d.tick.stale);status.textContent=healthy?(marketStale?'Paper attivo - ultimo feed MT5 in cache':'Paper attivo - '+roma(Math.floor(new Date(d.tick.timestamp).getTime()/1000))+' Roma'):'ATTENZIONE: motore paper fermo/stale';status.style.color=healthy?(marketStale?'#fbbf24':'#7dd3fc'):'#fb7185';
 for(const x of ['bid','ask','spread'])document.querySelector('#'+x).textContent=fmt(d.tick[x]);
 const inf=d.inference||{};probability('#pdown',inf.probability_down);probability('#pneutral',inf.probability_neutral);probability('#pup',inf.probability_up);
 const sig=document.querySelector('#signal');if(sig){sig.textContent=inf.candidate||'-';sig.className=inf.candidate==='BUY'?'buy':inf.candidate==='SELL'?'sell':'';}
 series.setData(d.candles);volume.setData(d.candles.map(x=>({time:x.time,value:x.volume,color:x.close>=x.open?'#1f8a78aa':'#b85766aa'})));const overlays=d.overlays||[];ema20.setData((overlays.find(x=>x.name==='EMA 20')||{data:[]}).data);ema100.setData((overlays.find(x=>x.name==='EMA 100')||{data:[]}).data);markerPlugin.setMarkers(d.markers);levels(d.levels);
 const select=document.querySelector('#strategy');if(!select.options.length){d.strategies.forEach(s=>select.add(new Option(s,s)));select.onchange=()=>{chosen=select.value;first=true;load()}}select.value=d.selected;
 const windowSelect=document.querySelector('#chart-window');if(windowSelect){windowSelect.value=chartWindow;windowSelect.onchange=()=>{chartWindow=windowSelect.value;first=true;load()}}
 document.querySelector('#cards').innerHTML=d.cards.map(c=>{let p=c.total_pnl||0,power=c.average_power==null?'N/D':fmt(c.average_power);return `<article class="card"><h3>${c.name}</h3><div class="pnl ${p>=0?'plus':'minus'}">${p>=0?'+':''}${fmt(p)} USD</div><div class="meta">aperto ${fmt(c.unrealized_pnl)} - chiuso ${fmt(c.realized_pnl)}<br>trade ${c.trades} - posizioni ${c.open_positions} - ${c.last_signal||'N/D'}<br>power media ${power}${c.powered_trades?` (${c.powered_trades} trade)`:''}</div></article>`}).join('');
 if(first){chart.timeScale().fitContent();first=false}
}
const endpoint=location.pathname==='/indicators'?'/api/indicators':'/api/snapshot';async function load(){try{const r=await fetch(endpoint+'?strategy='+encodeURIComponent(chosen)+'&window='+encodeURIComponent(chartWindow));if(!r.ok)throw new Error((await r.json()).detail);paint(await r.json())}catch(e){document.querySelector('#status').textContent='Errore feed: '+e.message}}
new ResizeObserver(()=>chart.applyOptions({width:host.clientWidth,height:host.clientHeight})).observe(host);load();setInterval(load,5000);
