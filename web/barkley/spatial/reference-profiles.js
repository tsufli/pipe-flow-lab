"use strict";
// Deterministic reference profiles for the Barkley spatial PDE dashboard.
// The decay-edge curve is obtained by finite-time basin bisection, and is
// deliberately labeled as a candidate rather than a verified exact orbit.
globalThis.BarkleyRef=(()=>{
 const sleep=()=>new Promise(resolve=>setTimeout(resolve,0));
 const physicsKey=p=>[p.r,p.delta,p.eps1,p.eps2,p.zeta,p.D,p.U0,p.Ub,p.L,Math.round(p.N)].join("|");
 function activity(q,dx){let mass=0,peak=0,extent=0;for(const y of q){const a=Math.max(0,y);mass+=a*dx;peak=Math.max(peak,a);if(a>.2)extent+=dx;}return {mass,peak,extent};}
 function center(q,L){let sx=0,cx=0,sum=0;for(let i=0;i<q.length;i++){const v=Math.max(0,q[i]),theta=2*Math.PI*(i+.5)/q.length;sum+=v;sx+=v*Math.sin(theta);cx+=v*Math.cos(theta);}if(sum<1e-5||Math.hypot(sx,cx)<.18*sum)return null;return (Math.atan2(sx,cx)/(2*Math.PI)+1)%1*L;}
 function align(data,fromQ,toQ,L){const src=center(fromQ,L),dest=center(toQ,L);if(src===null||dest===null)return null;const n=data.length,dx=L/n,shift=dest-src,out=new Float64Array(n);for(let j=0;j<n;j++){const pos=(j+.5)-shift/dx-.5,wrapped=((pos%n)+n)%n,k=Math.floor(wrapped),frac=wrapped-k;out[j]=data[k]*(1-frac)+data[(k+1)%n]*frac;}return out;}
 function rhs(q,u,c,dx){const N=q.length,dq=new Float64Array(N),du=new Float64Array(N);for(let i=0;i<N;i++){const im=(i+N-1)%N,ip=(i+1)%N,advSpeed=u[i]-c.zeta,adv=advSpeed>=0?advSpeed*(q[i]-q[im])/dx:advSpeed*(q[ip]-q[i])/dx;
 dq[i]=-adv+c.D*(q[im]-2*q[i]+q[ip])/(dx*dx)+q[i]*(c.r+u[i]-c.U0-(c.r+c.delta)*(q[i]-1)**2);
 const fl=.25*(u[im]**2+u[i]**2)-.5*Math.max(Math.abs(u[im]),Math.abs(u[i]))*(u[i]-u[im]);
 const fr=.25*(u[i]**2+u[ip]**2)-.5*Math.max(Math.abs(u[i]),Math.abs(u[ip]))*(u[ip]-u[i]);
 du[i]=-(fr-fl)/dx+c.eps1*(c.U0-u[i])+c.eps2*(c.Ub-u[i])*q[i];}
 return {dq,du};}
 function step(q,u,h,c,dx){const N=q.length,a=rhs(q,u,c,dx),q1=new Float64Array(N),u1=new Float64Array(N);for(let i=0;i<N;i++){q1[i]=q[i]+h*a.dq[i];u1[i]=u[i]+h*a.du[i];}
 const b=rhs(q1,u1,c,dx),qo=new Float64Array(N),uo=new Float64Array(N);for(let i=0;i<N;i++){qo[i]=q[i]+h*.5*(a.dq[i]+b.dq[i]);uo[i]=u[i]+h*.5*(a.du[i]+b.du[i]);}return {q:qo,u:uo};}
 async function evolve(initial,T,c,aborted,saveAt=null){const dx=c.L/Math.round(c.N),N=Math.round(c.N);let {q,u}=initial,t=0,steps=0,saved=null;q=q.slice();u=u.slice();while(t<T-1e-9){if(aborted())throw Error("cancelled");let speed=0;for(let j=0;j<N;j++)speed=Math.max(speed,Math.abs(u[j]),Math.abs(u[j]-c.zeta));
 const h=Math.min(T-t,.32*dx/Math.max(.05,speed),.22*dx*dx/c.D,.04),next=step(q,u,h,c,dx);q=next.q;u=next.u;t+=h;if(!Number.isFinite(q[0]+u[0]))throw Error("numerical instability");
 if(saveAt!==null&&saved===null&&t>=saveAt)saved={q:q.slice(),u:u.slice()};if(++steps%350===0)await sleep();}
 return {q,u,saved};}
 async function compute(c,progress=()=>{},aborted=()=>false){const N=Math.round(c.N),dx=c.L/N,q0=new Float64Array(N),u0=new Float64Array(N),seedWidth=Math.min(12,.3*c.L);
 for(let i=0;i<N;i++){const x=(i+.5)*dx,dist=Math.abs(x-c.L/2);q0[i]=1.65/(1+Math.exp(Math.min(100,(dist-seedWidth/2)/Math.max(dx,1))));u0[i]=c.U0;}
 progress("Relaxing a deterministic puff…");
 const tPuff=Math.min(380,Math.max(240,15/Math.min(c.eps1,c.eps2)));
 const pRun=await evolve({q:q0,u:u0},tPuff,c,aborted,tPuff-40),puff={q:pRun.q,u:pRun.u},m=activity(puff.q,dx),old=activity(pRun.saved.q,dx);
 if(m.mass<1||m.peak<.2||m.extent>.46*c.L||Math.abs(m.mass-old.mass)/Math.max(m.mass,1)>.12)throw Error("no stable, localized deterministic puff was found");
 let lo=0,hi=1;const tClass=Math.min(280,Math.max(155,14/Math.min(c.eps1,c.eps2)));
 for(let k=0;k<21;k++){const alpha=(lo+hi)/2,qi=new Float64Array(N),ui=new Float64Array(N);
 for(let j=0;j<N;j++){qi[j]=alpha*puff.q[j];ui[j]=c.U0+alpha*(puff.u[j]-c.U0);}
 const run=await evolve({q:qi,u:ui},tClass,c,aborted),am=activity(run.q,dx).mass;
 if(am>.6*m.mass)hi=alpha;else lo=alpha;if(k%3===0)progress("Tracking decay edge (bisection "+(k+1)+"/21)…");}
 const alpha=(lo+hi)/2,qi=new Float64Array(N),ui=new Float64Array(N);
 for(let j=0;j<N;j++){qi[j]=alpha*puff.q[j];ui[j]=c.U0+alpha*(puff.u[j]-c.U0);}
 const tEdge=Math.min(65,Math.max(32,.25*tClass));
 const result=await evolve({q:qi,u:ui},tEdge+15,c,aborted,tEdge),edge=result.saved,edgeStats=activity(edge.q,dx),later=activity(result.q,dx);
 if(edgeStats.mass<.07*m.mass||edgeStats.mass>1.07*m.mass||edgeStats.peak<.06||Math.abs(later.mass-edgeStats.mass)/Math.max(edgeStats.mass,1e-8)>.28||center(edge.q,c.L)===null)throw Error("no persistent localized decay-edge candidate was resolved");
 return {puff,edge,key:physicsKey(c),alpha,info:"Finite-time deterministic edge-tracking candidate (not a certified traveling-wave solution)",puffMass:m.mass,edgeMass:edgeStats.mass};}
 return {compute,align,center,physicsKey};
})();