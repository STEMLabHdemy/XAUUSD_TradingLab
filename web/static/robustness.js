const money=value=>Number(value||0).toLocaleString('it-IT',{minimumFractionDigits:2,maximumFractionDigits:2});
const cell=(value,good)=>`<td class="${good===undefined?'':good?'plus':'minus'}">${value}</td>`;
async function load(){
  try{
    const response=await fetch('/api/research/adaptive-robustness',{cache:'no-store'}),data=await response.json();
    if(!response.ok)throw Error(data.detail);
    const status=data.status||{},ranking=data.ranking||[],runs=data.runs||[];
    document.querySelector('#status').textContent=`${status.message||'attesa'} · ${status.completed||0}/${status.total||24}`;
    document.querySelector('#status').style.color=status.message==='studio robustezza completato'?'#34d399':'#60a5fa';
    document.querySelector('#summary').innerHTML=[['Replay completati',`${status.completed||0} / ${status.total||24}`],['Blocchi',data.days_per_fold?`${data.days_per_fold} giorni × 4`:'—'],['Costi testati',data.cost_multipliers?data.cost_multipliers.map(x=>`${x}×`).join(' · '):'—']].map(x=>`<article><span>${x[0]}</span><b>${x[1]}</b></article>`).join('');
    document.querySelector('#ranking').innerHTML=ranking.map(row=>`<tr>${cell(`${row.label} <small>(${row.source_candidate})</small>`)}${cell(`${row.profitable_folds} / 4`)}${cell(row.median_pnl==null?'—':`${row.median_pnl>=0?'+':''}${money(row.median_pnl)} USD`,row.median_pnl>=0)}${cell(row.worst_pnl==null?'—':`${row.worst_pnl>=0?'+':''}${money(row.worst_pnl)} USD`,row.worst_pnl>=0)}${cell(row.median_pf==null?'—':money(row.median_pf),row.median_pf>1)}${cell(row.median_stress_pnl==null?'—':`${row.median_stress_pnl>=0?'+':''}${money(row.median_stress_pnl)} USD`,row.median_stress_pnl>=0)}${cell(row.grade,row.grade==='DA APPROFONDIRE')}</tr>`).join('')||'<tr><td colspan="7">Primo replay in corso…</td></tr>';
    document.querySelector('#runs').innerHTML=runs.sort((a,b)=>a.key.localeCompare(b.key)).map(row=>{const m=row.adaptive;return `<tr>${cell(row.challenger_id)}${cell(row.fold.id)}${cell(new Date(row.fold.end).toLocaleDateString('it-IT'))}${cell(`${row.cost_multiplier}×`)}${cell(`${m.net_pnl>=0?'+':''}${money(m.net_pnl)} USD`,m.net_pnl>=0)}${cell(money(m.profit_factor),m.profit_factor>1)}${cell(m.trades)}${cell(`${(m.max_drawdown*100).toFixed(2)}%`)}</tr>`}).join('')||'<tr><td colspan="8">Primo replay in corso…</td></tr>';
  }catch(error){document.querySelector('#status').textContent=`Errore: ${error.message}`;document.querySelector('#status').style.color='#fb7185'}
}
load();setInterval(load,3000);
