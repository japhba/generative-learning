const $ = id => document.getElementById(id);
const config = {responsive:true, displaylogo:false, toImageButtonOptions:{format:'svg'}};
const mean = x => x.reduce((a,b)=>a+b,0)/x.length;
const sem = x => {const m=mean(x);return Math.sqrt(x.reduce((a,b)=>a+(b-m)**2,0)/(x.length-1)/x.length);};
const median = x => {const y=[...x].sort((a,b)=>a-b),n=y.length;return (y[Math.floor((n-1)/2)]+y[Math.ceil((n-1)/2)])/2;};
const color = a => DATA.colors[a] || '#777777';
const labels = {kl:'Population KL (nats)',validity:'Population validity',root_mi:'Normalized root information',branch_mi:'Normalized branch information',forward_objective:'Forward joint KL (nats)',objective:'Reverse joint KL (nats)',effective_modes:'Effective number of valid modes',importance_ess:'ESS / batch size for ancestral proposal'};
function layout(title,xlabel,ylabel){return {title:{text:title},xaxis:{title:{text:xlabel},automargin:true},yaxis:{title:{text:ylabel},automargin:true},showlegend:false,hovermode:'closest',paper_bgcolor:'white',plot_bgcolor:'white',autosize:true};}
function draw(id,traces,l){
  let legend=$(id+'-legend');
  if(!legend){legend=document.createElement('div');legend.id=id+'-legend';legend.className='controls';$(id).before(legend);}
  legend.replaceChildren();
  for(const t of traces.filter(t=>t.name&&t.showlegend!==false)){
    const button=document.createElement('button');button.textContent=t.name.replace(/ · seed \d+$/,'');button.type='button';
    const group=t.legendgroup||t.name,ix=traces.map((v,i)=>(v.legendgroup||v.name)===group?i:-1).filter(i=>i>=0);
    let shown=t.visible!=='legendonly'&&t.visible!==false;
    const paint=()=>{button.setAttribute('aria-pressed',String(shown));button.style.opacity=shown?'1':'.45';button.style.color=(t.line&&t.line.color)||(t.marker&&t.marker.color)||'#24292e';};
    button.addEventListener('click',()=>{shown=!shown;Plotly.restyle(id,{visible:shown},ix);paint();});paint();legend.append(button);
  }
  Plotly.react(id,traces,l,config);
}
function band(traces,x,values,name,c,show=true){
  const m=x.map((_,i)=>mean(values.map(v=>v[i]))),s=x.map((_,i)=>sem(values.map(v=>v[i])));
  traces.push({x,y:m.map((v,i)=>v-s[i]),mode:'lines',line:{color:c},opacity:.12,hoverinfo:'skip',showlegend:false,legendgroup:name});
  traces.push({x,y:m.map((v,i)=>v+s[i]),mode:'lines',line:{color:c},opacity:.12,fill:'tonexty',hoverinfo:'skip',showlegend:false,legendgroup:name});
  traces.push({x,y:m,mode:'lines',name,line:{color:c},legendgroup:name,showlegend:show});
}
function learning(){
  const rows=DATA.studies[$('study').value],metric=$('metric').value,view=$('curve-view').value,traces=[];
  const arms=Object.keys(rows[0].curves);
  for(const a of arms){
    const x=rows[0].curves[a].step.map(t=>t+1),ys=rows.map(r=>r.curves[a][metric]),name=DATA.names[a];
    if(view==='mean') band(traces,x,ys,name,color(a));
    if(view==='median') traces.push({x,y:x.map((_,i)=>median(ys.map(y=>y[i]))),mode:'lines',name,line:{color:color(a)}});
    if(view==='seeds') rows.forEach((r,i)=>traces.push({x,y:ys[i],mode:'lines',name:name+' · seed '+r.seed,legendgroup:a,showlegend:i===0,line:{color:color(a)},opacity:.3}));
  }
  const l=layout('Learning curves','Continuation updates + 1',labels[metric]);l.xaxis.type='log';l.xaxis.dtick=1;
  draw('learning',traces,l);
}
function world(){
  const z=Number($('world-z').value),u=Number($('world-u').value),v=Number($('world-v').value);
  const x=[u,z*u,z,v,z*v,z];if($('world-error').checked)x[5]*=-1;
  $('world-bits').replaceChildren(...x.map(v=>{const d=document.createElement('div');d.className='bit';d.textContent=v>0?'+1':'−1';return d;}));
  const verdict=v=>v===1?'passes':'fails';
  $('world-checks').textContent='Left relation: '+verdict(x[0]*x[1]*x[2])+' · Right relation: '+verdict(x[3]*x[4]*x[5])+' · Shared-root relation: '+verdict(x[2]*x[5]);
}
const strings=Array.from({length:64},(_,i)=>i.toString(2).padStart(6,'0').replaceAll('0','−').replaceAll('1','+'));
const rows=DATA.studies.exploration_local;
for(const r of rows){const o=document.createElement('option');o.value=r.seed;o.textContent='Seed '+r.seed;$('inspect-seed').append(o);}
function softmax(x){const m=Math.max(...x),e=x.map(v=>Math.exp(v-m)),s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s);}
function inspect(){
  const r=rows.find(x=>x.seed===Number($('inspect-seed').value)),view=$('inspect-view').value,a=$('inspect-arm').value;
  if(view==='distribution'){
    const ix=r.valid_indices,x=ix.map(i=>strings[i]).concat(['Invalid']),traces=[];
    for(const arm of ['forward_explore','forward','watch','fresh','ae','critic']){
      const p=r.distributions[arm],y=ix.map(i=>p[i]);y.push(1-mean(y)*8);
      traces.push({x,y,type:'bar',name:arm==='critic'?'Learned constraint distribution':DATA.names[arm],marker:{color:arm==='critic'?'#9d99ad':color(arm)},visible:['forward','ae','critic'].includes(arm)?'legendonly':true});
    }
    const empirical=ix.map(i=>r.observed_indices.filter(j=>j===i).length/r.observed_indices.length).concat([0]);
    traces.push({x,y:empirical,type:'bar',name:'Empirical demonstrations',visible:'legendonly',marker:{color:'#999999'}});
    traces.push({x,y:ix.map(()=>1/8).concat([0]),mode:'markers',name:'True population',marker:{color:'#202326'}});
    const l=layout('Final generation and coverage · seed '+r.seed,'Observation','Probability');l.barmode='group';
    draw('distribution',traces,l);
  }else if(view==='emissions'){
    const d=r.parameters[a][1],z=[],y=[];
    for(let b=0;b<2;b++)for(let h=0;h<4;h++){z.push(Array.from({length:6},(_,j)=>Math.floor(j/3)===b?Math.tanh(d[b][h][j%3]/2):null));y.push((b===0?'Left':'Right')+' state '+h);}
    draw('distribution',[{type:'heatmap',x:['Leaf 1','Leaf 2','Leaf 3','Leaf 4','Leaf 5','Leaf 6'],y,z,colorscale:'RdBu',zmin:-1,zmax:1,colorbar:{title:'Mean sign'}}],layout('Decoder emissions · seed '+r.seed,'Observation unit','Branch latent'));
  }else{
    const t=r.parameters[a][0],z=[0,1].map(root=>softmax(t[0][root]).concat(softmax(t[1][root]))),x=[0,1].flatMap(b=>Array.from({length:4},(_,h)=>(b===0?'Left ':'Right ')+h));
    draw('distribution',[{type:'heatmap',x,y:['Root state 0','Root state 1'],z,colorscale:'Blues',zmin:0,zmax:1,colorbar:{title:'Probability'}}],layout('Root-to-branch conditionals · seed '+r.seed,'Branch state','Root state'));
  }
}
function depth(){
  const traces=[],x=rows[0].curves.forward_explore.step.map(t=>t+1);
  band(traces,x,rows.map(r=>r.curves.forward_explore.branch_mi),'Branch causes','#b37932');
  band(traces,x,rows.map(r=>r.curves.forward_explore.root_mi),'Shared root cause','#196a85');
  const l=layout('Latent information during offline practice','Practice updates + 1','Mutual information / true-cause entropy');l.xaxis.type='log';l.xaxis.dtick=1;l.yaxis.range=[0,1];
  draw('depth',traces,l);
}
function sensitivity(){
  const coupling=Number($('coupling').value),metric=$('sensitivity-metric').value,sub=DATA.sensitivity.filter(r=>r.coupling===coupling),ns=[...new Set(sub.map(r=>r.n))].sort((a,b)=>a-b),traces=[];
  for(const a of ['forward_explore','watch','fresh','ae']){
    const ys=ns.map(n=>sub.filter(r=>r.n===n).map(r=>r.end[a][metric]));
    traces.push({x:ns,y:ys.map(mean),error_y:{type:'data',array:ys.map(sem),visible:true},mode:'lines+markers',name:DATA.names[a],line:{color:color(a)}});
  }
  if(metric==='kl')traces.push({x:ns,y:ns.map(n=>mean(sub.filter(r=>r.n===n).map(r=>r.critic_kl))),mode:'lines',name:'Learned constraint model',line:{color:'#777777',dash:'dot'}});
  const l=layout('Sample-size sensitivity · coupling '+coupling,'Number of demonstrations',labels[metric]);l.xaxis.type='log';l.xaxis.tickvals=ns;
  draw('sensitivity',traces,l);
}
function budget(){
  const metric=$('budget-metric').value,rs=DATA.allocation,x=[...new Set(rs.map(r=>r.total))].sort((a,b)=>a-b),y=[...new Set(rs.map(r=>r.fraction))].sort((a,b)=>a-b);
  const cells=y.map(f=>x.map(n=>rs.filter(r=>r.total===n&&r.fraction===f).map(r=>r.end[metric]))),z=cells.map(row=>row.map(mean));
  const custom=cells.map(row=>row.map(v=>[sem(v),v.length]));
  const l=layout('Watching / practice allocation','Sample presentations to the networks','Fraction assigned to offline practice');l.xaxis.type='category';
  Plotly.react('budget',[{type:'heatmap',x:x.map(String),y,z,customdata:custom,colorscale:metric==='kl'?'Viridis':'Blues',reversescale:metric==='kl',colorbar:{title:labels[metric]},hovertemplate:'Presentations: %{x}<br>Practice fraction: %{y}<br>Mean: %{z:.3f}<br>SE: %{customdata[0]:.3f}<br>Seeds: %{customdata[1]}<extra></extra>'}],l,config);
}
for(const id of ['study','metric','curve-view'])$(id).addEventListener('change',learning);
for(const id of ['world-z','world-u','world-v','world-error'])$(id).addEventListener('change',world);
for(const id of ['inspect-seed','inspect-view','inspect-arm'])$(id).addEventListener('change',inspect);
for(const id of ['coupling','sensitivity-metric'])$(id).addEventListener('change',sensitivity);
$('budget-metric').addEventListener('change',budget);
world();learning();inspect();depth();sensitivity();budget();
document.documentElement.dataset.reportReady='true';
