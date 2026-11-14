"use strict";
// Profile viewer for saved two-start Barkley instanton datasets.
function sampleIndices(){
 const n=frames().length,k=Number($('curveCount').value);
 return Array.from({length:k},(_,i)=>Math.round((i+1)*(n-1)/(k+1)));
}
function geometricInitial(){return field==='q' && $('initialKind').value==='reference';}
function pathFrames(kind,id){
 return kind==='initial'
  ? (geometricInitial()?current.seeds[id].reference.frames:current.seeds[id].stages.initial.frames)
  : current.seeds[id].stages[stage].frames;
}
function pathProfile(kind,id,index){
 const f=pathFrames(kind,id)[index],y=f[field];
 if(!Array.isArray(y))return [];
 return $('followPeak').checked?rotateForDisplay(y,f,current):y;
}
function endpointProfile(kind){
 const f=current[kind];
 return $('followPeak').checked?rotateForDisplay(f[field],f,current):f[field];
}
function profileBasis(){
 const x=xOf(current);
 return [curve(x,endpointProfile('puff'),'Stable puff','#9eb0c6','dot'),
         curve(x,endpointProfile('edge'),'Decay edge','#c7a0ed','dash')];
}
function movingTrace(kind,id,name,color,dash='solid'){
 const tr=curve(xOf(current),pathProfile(kind,id,tidx),name,color,dash);
 tr.line.width=3.5;return tr;
}
function profileOverlay(kind){
 const out=profileBasis(),indices=sampleIndices(),x=xOf(current);
 indices.forEach((i,j)=>{
   const pos=j/(indices.length-1||1),t=pathFrames(kind,seed)[i].t;
   const tr=curve(x,pathProfile(kind,seed,i),'Intermediate t='+fmt(t,1),'hsl('+(190-155*pos)+',80%,65%)');
   tr.line.width=1.75;tr.opacity=.72;tr.showlegend=false;out.push(tr);
 });
 out.push(movingTrace(kind,seed,'Selected time',kind==='initial'?'#66dbed':'#f4ad63'));
 if($('compare').checked){
   const other=seed==='linear'?'reverse':'linear';
   const tr=movingTrace(kind,other,'Other initialization at selected time','#eb82ae','dash');
   tr.line.width=2.2;out.push(tr);
 }
 return out;
}
function deformationTraces(){
 const lines=profileBasis();
 lines.push(movingTrace('initial',seed,'Initial path','#66dbed'));
 lines.push(movingTrace('optimized',seed,'Optimized path','#f4ad63'));
 if($('compare').checked){
   const alt=seed==='linear'?'reverse':'linear';
   for(const kind of ['initial','optimized']){
     const tr=movingTrace(kind,alt,'Other start · '+kind,'#eb82ae','dash');
     tr.line.width=2;tr.opacity=kind==='initial'?.65:.95;lines.push(tr);
   }
 }
 return lines;
}
function profilesLayout(){
 return {xaxis:{title:'x − initial puff peak (comoving frame)',gridcolor:PALETTE.axis,range:[-46,72]},
  yaxis:{title:field,gridcolor:PALETTE.axis},
  margin:{l:52,r:15,t:12,b:52},legend:{orientation:'h',y:-.23,font:{size:10}}};
}
function drawProfiles(){
 const reference=geometricInitial();
 $('initialHeading').textContent=reference
  ? 'Initial path · '+(seed==='linear'?'linear interpolation':'reversed unstable branch')
  : 'Initial control · PDE rollout';
 $('initialNote').textContent=reference
  ? 'Geometric q reference only: these snapshots do not generally satisfy the coupled PDE. Select the initial-control rollout to see a valid forward evolution.'
  : 'Actual PDE rollout under the scaled control derived from the geometric reference. It need not follow that reference exactly.';
 $('optimizedHeading').textContent=stage==='initial'
  ? 'Before optimization · initial-control rollout'
  : 'Optimized path · endpoint penalty ρ = '+stage;
 $('optimizedNote').textContent=stage==='initial'
  ? 'Select a non-initial checkpoint to see the optimized trajectory.'
  : 'Snapshots of the controlled PDE; the terminal match is penalized, not imposed exactly.';
 $('movieTime').textContent='t = '+fmt(frames()[tidx].t,1)+' / '+fmt(current.params.T,1);
 draw('initialPlot',profileOverlay('initial'),profilesLayout());
 draw('optimizedPlot',profileOverlay('optimized'),profilesLayout());
 draw('deformPlot',deformationTraces(),profilesLayout());
}
function updateMovingProfiles(){
 if(view!=='fields'||!hasData())return;
 const i=2+sampleIndices().length;
 $('movieTime').textContent='t = '+fmt(frames()[tidx].t,1)+' / '+fmt(current.params.T,1);
 for(const kind of ['initial','optimized']){
   const id=kind==='initial'?'initialPlot':'optimizedPlot';
   Plotly.restyle($(id),{y:[pathProfile(kind,seed,tidx)]},[i]);
   if($('compare').checked){
     const alt=seed==='linear'?'reverse':'linear';
     Plotly.restyle($(id),{y:[pathProfile(kind,alt,tidx)]},[i+1]);
   }
 }
 Plotly.restyle($('deformPlot'),{y:[pathProfile('initial',seed,tidx)]},[2]);
 Plotly.restyle($('deformPlot'),{y:[pathProfile('optimized',seed,tidx)]},[3]);
 if($('compare').checked){
   const other=seed==='linear'?'reverse':'linear';
   Plotly.restyle($('deformPlot'),{y:[pathProfile('initial',other,tidx)]},[4]);
   Plotly.restyle($('deformPlot'),{y:[pathProfile('optimized',other,tidx)]},[5]);
 }
}
