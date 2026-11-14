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
 const f=pathFrames(kind,id)[index];
 const y=f[field];
 if(!Array.isArray(y))return [];
 return $("followPeak").checked?rotateForDisplay(y,f,current):y;
}
function endpointProfile(kind){
 const obj=current[kind],y=obj[field];
 return $("followPeak").checked?rotateForDisplay(y,obj,current):y;
}
function endpointTraces(){
 const x=xOf(current);
 return [
  {type:"scatter",mode:"lines",x,y:endpointProfile("puff"),name:"Stable puff",
    line:{color:PROFILE_COLORS.puff,width:4,dash:"solid"},hovertemplate:"x=%{x:.1f}<br>puff: %{y:.5f}<extra></extra>"},
  {type:"scatter",mode:"lines",x,y:endpointProfile("edge"),name:"Decay edge",
    line:{color:PROFILE_COLORS.edge,width:4,dash:"solid"},hovertemplate:"x=%{x:.1f}<br>edge: %{y:.5f}<extra></extra>"}
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
function profileLayout(showLegend=true){
 return {
  xaxis:{title:"x − initial puff peak (comoving)",gridcolor:PALETTE.axis,range:[-42,63]},
  yaxis:{title:field,gridcolor:PALETTE.axis,range:commonYRange()},
  margin:{l:52,r:15,t:9,b:54},
  showlegend:showLegend,legend:{orientation:"h",y:-0.23,font:{size:11}},
  hovermode:"closest",uirevision:"selected-profile"
 };
}
function interpolationTraces(kind){
 const x=xOf(current),samples=sampleIndices();
 const shades=["#778da6","#8da0b8","#aabacc","#c5d0de","#e1e6ef"];
 return samples.map((i,j)=>{
  const t=pathFrames(kind)[i].t;
  // Neutral, thin, solid samples; the endpoints are added LAST and stand out.
  return {type:"scatter",mode:"lines",x,y:pathProfile(kind,i),
    name:"Intermediate states",showlegend:j===0,
    line:{color:shades[Math.round(4*j/Math.max(1,samples.length-1))],width:1.8,dash:"solid"},
    opacity:0.82,
    hovertemplate:"t="+fmt(t,1)+"<br>x=%{x:.1f}<br>"+field+"=%{y:.5f}<extra>Intermediate profile</extra>"};
 });
}
function staticTraces(kind){
 return [...interpolationTraces(kind),...endpointTraces()];
}
function movingTrace(kind){
 return {type:"scatter",mode:"lines",x:xOf(current),y:pathProfile(kind,tidx),
  name:kind==="initial"?"Initial path":"Optimized path",
  line:{color:kind==="initial"?PROFILE_COLORS.initial:PROFILE_COLORS.optimized,
        width:4.5,dash:"solid"},
  hovertemplate:"x=%{x:.1f}<br>"+field+"=%{y:.5f}<extra>t="+fmt(frames()[tidx].t,1)+"</extra>"};
}
function movieTraces(kind){
 // Keep the two reference profiles secondary in the movies.
 const fixed=endpointTraces().map(t=>({
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
 Plotly.restyle($("initialMovie"),{y:[pathProfile("initial",tidx)]},[2]);
 Plotly.restyle($("optimizedMovie"),{y:[pathProfile("optimized",tidx)]},[2]);
}
