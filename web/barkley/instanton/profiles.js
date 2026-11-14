"use strict";
// The two profile overlays are deliberately static. Only the TWO separate
// deformation movies respond to playback; no heatmaps or changing upper curves.
const PROFILE_COLORS={
  puff:"#46cfff",
  edge:"#ffb068",
  intermediate:"#a3b4c9",
  initial:"#69f0df",
  optimized:"#f9cf75"
};
function sampleIndices(){
 const n=frames().length,count=Number($("curveCount").value);
 return Array.from({length:count},(_,i)=>Math.round((i+1)*(n-1)/(count+1)));
}
function geometricInitial(){return field==="q" && $("initialKind").value==="reference";}
function pathFrames(kind,id=seed){
 return kind==="initial"
  ? (geometricInitial()?current.seeds[id].reference.frames:current.seeds[id].stages.initial.frames)
  : current.seeds[id].stages[stage].frames;
}
function pathProfile(kind,index,id=seed){
 const frame=pathFrames(kind,id)[index];
 return frame[field]; // Preserve the stored PDE field without shifting or peak registration.
}
function endpointProfile(kind){return current[kind][field];}
// Stored solutions were integrated with z=x_lab-c_puff*t.
// Transform the abscissa, NEVER the data arrays, back to one fixed lab origin.
function labXAt(time,speed=current.puff.c){
 return xOf(current).map(x=>x+speed*time);
}
function pathX(kind,index,id=seed){
 return labXAt(pathFrames(kind,id)[index].t,current.puff.c);
}
// Reference waves are independent freely propagating solutions; edge.c differs from puff.c.
function endpointTraces(time=0){
 return [
  {type:"scatter",mode:"lines",x:labXAt(time,current.puff.c),y:endpointProfile("puff"),name:"Stable puff",
    line:{color:PROFILE_COLORS.puff,width:4,dash:"solid"},hovertemplate:"x_lab=%{x:.1f}<br>puff: %{y:.5f}<extra></extra>"},
  {type:"scatter",mode:"lines",x:labXAt(time,current.edge.c),y:endpointProfile("edge"),name:"Decay edge",
    line:{color:PROFILE_COLORS.edge,width:4,dash:"solid"},hovertemplate:"x_lab=%{x:.1f}<br>edge: %{y:.5f}<extra></extra>"}
 ];
}
function commonYRange(){
 let lo=Infinity,hi=-Infinity;
 const accumulate=arr=>{
  if(!Array.isArray(arr))return;
  for(const v of arr){if(Number.isFinite(v)){lo=Math.min(lo,v);hi=Math.max(hi,v);}}
 };
 accumulate(endpointProfile("puff"));accumulate(endpointProfile("edge"));
 for(const kind of ["initial","optimized"])
  for(const frame of pathFrames(kind))accumulate(frame[field]);
 if(!Number.isFinite(lo)||!Number.isFinite(hi))return undefined;
 const span=Math.max(0.15,hi-lo);
 return [lo-0.08*span,hi+0.08*span];
}
function labDisplayRange(){
 // Determine one fixed view window covering all saved peaks in the LAB frame.
 // Peak positions are used to choose axis limits ONLY, not to align the profiles.
 const base=xOf(current),endTime=current.params.T;
 const positions=[
   base[current.puff.q.indexOf(Math.max(...current.puff.q))],
   base[current.edge.q.indexOf(Math.max(...current.edge.q))],
   base[current.puff.q.indexOf(Math.max(...current.puff.q))]+current.puff.c*endTime,
   base[current.edge.q.indexOf(Math.max(...current.edge.q))]+current.edge.c*endTime
 ];
 for(const kind of ["initial","optimized"])
  for(const f of pathFrames(kind)){
   const i=f.q.indexOf(Math.max(...f.q));
   positions.push(base[i]+current.puff.c*f.t);
  }
 return [Math.min(...positions)-33,Math.max(...positions)+36];
}
function profileLayout(showLegend=true){
 return {
  xaxis:{title:"x_lab (fixed origin: initial puff peak at t = 0)",gridcolor:PALETTE.axis,range:labDisplayRange()},
  yaxis:{title:field,gridcolor:PALETTE.axis,range:commonYRange()},
  margin:{l:52,r:15,t:9,b:54},
  showlegend:showLegend,legend:{orientation:"h",y:-0.23,font:{size:11}},
  hovermode:"closest",uirevision:"selected-profile"
 };
}
function interpolationTraces(kind){
 const samples=sampleIndices();
 const shades=["#778da6","#8da0b8","#aabacc","#c5d0de","#e1e6ef"];
 return samples.map((i,j)=>{
  const t=pathFrames(kind)[i].t;
  // Neutral, thin, solid samples; the endpoints are added LAST and stand out.
  return {type:"scatter",mode:"lines",x:pathX(kind,i),y:pathProfile(kind,i),
    name:"Intermediate states",showlegend:j===0,
    line:{color:shades[Math.round(4*j/Math.max(1,samples.length-1))],width:1.8,dash:"solid"},
    opacity:0.82,
    hovertemplate:"t="+fmt(t,1)+"<br>x_lab=%{x:.1f}<br>"+field+"=%{y:.5f}<extra>Intermediate profile</extra>"};
 });
}
function staticTraces(kind){
 return [...interpolationTraces(kind),...endpointTraces()];
}
function movingTrace(kind){
 return {type:"scatter",mode:"lines",x:pathX(kind,tidx),y:pathProfile(kind,tidx),
  name:kind==="initial"?"Initial path":"Optimized path",
  line:{color:kind==="initial"?PROFILE_COLORS.initial:PROFILE_COLORS.optimized,
        width:4.5,dash:"solid"},
  hovertemplate:"x_lab=%{x:.1f}<br>"+field+"=%{y:.5f}<extra>t="+fmt(frames()[tidx].t,1)+"</extra>"};
}
function movieTraces(kind){
 // Keep the two reference profiles secondary in the movies.
 const fixed=endpointTraces(frames()[tidx].t).map(t=>({
   ...t,opacity:0.55,showlegend:true,line:{...t.line,width:2.6}
 }));
 return [...fixed,movingTrace(kind)];
}
function drawProfiles(){
 const reference=geometricInitial();
 $("initialHeading").textContent=reference
  ? "Initial path · "+(seed==="linear"?"linear interpolation":"reversed unstable branch")
  : "Initial-control rollout · before minimization";
 $("initialNote").textContent=reference
  ? "Stored q-only geometric reference (not a physical PDE trajectory). The sampled intermediate shapes are shown in gray, with the endpoints emphasized."
  : "Unoptimized forward PDE evolution under the scaled initial control; not the geometric reference.";
 $("optimizedHeading").textContent=stage==="initial"
  ? "Before optimization · initial-control rollout"
  : "Optimized path · penalty ρ = "+stage;
 $("optimizedNote").textContent=stage==="initial"
  ? "Choose a non-initial checkpoint to examine the optimization result."
  : "Actual controlled-PDE profiles; endpoint matching is soft, not exact.";
 $("initialMovieHeading").textContent=reference
  ? "Movie 1 · "+(seed==="linear"?"linear interpolation":"reversed unstable branch")
  : "Movie 1 · initial-control rollout";
 $("optimizedMovieHeading").textContent=stage==="initial"
  ? "Movie 2 · before optimization"
  : "Movie 2 · optimized trajectory";
 draw("initialPlot",staticTraces("initial"),profileLayout());
 draw("optimizedPlot",staticTraces("optimized"),profileLayout());
 draw("initialMovie",movieTraces("initial"),profileLayout());
 draw("optimizedMovie",movieTraces("optimized"),profileLayout());
 updateMovieLabels();
}
function updateMovieLabels(){
 const time="t = "+fmt(frames()[tidx].t,1)+" / "+fmt(current.params.T,1);
 $("initialMovieTime").textContent=time;
 $("optimizedMovieTime").textContent=time;
}
function updateMovingProfiles(){
 if(view!=="fields"||!hasData())return;
 updateMovieLabels();
 // The upper initialPlot and optimizedPlot never change during playback.
 const t=frames()[tidx].t;
 const puffX=labXAt(t,current.puff.c),edgeX=labXAt(t,current.edge.c);
 for(const kind of ["initial","optimized"]){
  // Move both free-traveling references and the controlled/geometric trajectory.
  // Reuse their immutable field samples: no artificial centering at each frame.
  Plotly.restyle($(kind==="initial"?"initialMovie":"optimizedMovie"),{
   x:[puffX,edgeX,pathX(kind,tidx)],
   y:[endpointProfile("puff"),endpointProfile("edge"),pathProfile(kind,tidx)]
  },[0,1,2]);
 }
}
