"use strict";
// Regression tests for the instanton viewer's fixed lab-frame coordinate.
// Node built-ins only: no browser, no JAX, and no numerical recomputation.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const root = path.resolve(__dirname, "..");
const source = fs.readFileSync(path.join(root, "web/barkley/instanton/profiles.js"),"utf8");

const x0 = [-2,0,2,4];
const q0 = [0,1,0,0], q1 = [0,.8,.2,0], q2 = [0,.5,.5,0];
const u0 = [2,1.9,2,2], u1 = [2,1.8,1.9,2], u2 = [2,1.75,1.8,2];
const makeFrames = () => [
 {t:0,q:q0,u:u0},{t:1.5,q:q1,u:u1},{t:3,q:q2,u:u2}
];
const stages = {initial:{frames:makeFrames()},"100":{frames:makeFrames()}};
const current = {
 params:{N:4,dx:2,T:3},
 puff:{c:2,q:q0,u:u0},
 edge:{c:.5,q:[0,0,1,0],u:u2},
 seeds:{linear:{reference:{frames:makeFrames()},stages}}
};
const elements={
 initialKind:{value:"reference"},curveCount:{value:"1"},
 initialMovieTime:{textContent:""},optimizedMovieTime:{textContent:""}
};
const calls=[];
const context=vm.createContext({
  current,seed:"linear",field:"q",stage:"100",tidx:2,view:"fields",
  $:id=>elements[id],hasData:()=>true,
  xOf:()=>x0,frames:()=>stages["100"].frames,
  fmt:(x)=>String(x),
  draw:()=>{},
  Plotly:{restyle:(el,data,ids)=>calls.push({el,data,ids})},
  PALETTE:{axis:"#ccc"}
});
vm.runInContext(source,context,{filename:"profiles.js"});
const get = expression => vm.runInContext(expression,context);
const plain = arr => Array.from(arr);
assert.deepEqual(plain(get("labXAt(3)")),[4,6,8,10],
 "Lab coordinates must advance by c_puff*t");
assert.deepEqual(plain(get("labXAt(3,current.edge.c)")),[-.5,1.5,3.5,5.5],
 "The independent edge reference must propagate at its own velocity");
assert.deepEqual(plain(get("pathX('optimized',2)")),[4,6,8,10],
 "An optimized snapshot must use its actual lab position");
assert.deepEqual(plain(get("pathProfile('optimized',2)")),q2,
 "The profile ordinates must NOT be shifted or recentered");
assert.deepEqual(plain(get("staticTraces('initial')[0].x")),[1,3,5,7],
 "The static interpolation at t=1.5 must use the lab frame");
assert.deepEqual(plain(get("staticTraces('initial').at(-1).x")),x0,
 "The static edge reference is plotted at t=0");
assert.deepEqual(plain(get("movieTraces('optimized')[0].x")),[4,6,8,10]);
assert.deepEqual(plain(get("movieTraces('optimized')[1].x")),[-.5,1.5,3.5,5.5]);
get("updateMovingProfiles()");
assert.equal(calls.length,2, "Only the two movies should be updated");
for(const call of calls){
 assert.deepEqual(plain(call.ids),[0,1,2]);
 assert.deepEqual(plain(call.data.x[0]),[4,6,8,10]);
 assert.deepEqual(plain(call.data.x[1]),[-.5,1.5,3.5,5.5]);
 assert.deepEqual(plain(call.data.x[2]),[4,6,8,10]);
 assert.deepEqual(plain(call.data.y[2]),q2);
}
assert.ok(!source.includes("followPeak"));
assert.ok(!source.includes("rotateForDisplay"));
console.log("Lab coordinates, unmodified profiles and synchronized movie positions: PASS");
