#include "ogmaneo/Actor.h"
#include "ogmaneo/Hierarchy.h"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace ogmaneo;
int main(){ComputeSystem cs;ComputeSystem::setNumThreads(1);cs.rng.seed(11);Actor a;Actor::VisibleLayerDesc d;d.size=Int3(1,1,2);d.radius=0;a.initRandom(cs,Int3(1,1,3),16,{d});a.minSteps=0;a.historyIters=1;a.setPrior({1.f,1.f,0.f});IntBuffer x(1,0),action(1,0);
 for(int t=0;t<8;t++){a.setBehaviorProbPrev({.2f,.3f,.5f});a.step(cs,{&x},&action,0.f,true,false,.9f,false);for(float r:a.getOutputResidual())assert(r==0.f);}
 a.clearHistory();assert(a.getHistorySize()==0);float rewards[]={0,0,1,0};float discounts[]={1,.8f,.7f,.6f};
 for(int t=0;t<4;t++){x[0]=t%2;action[0]=t%3;FloatBuffer p={.1f+.01f*t,.3f,.6f-.01f*t};a.setBehaviorProbPrev(p);a.step(cs,{&x},&action,rewards[t],true,false,discounts[t],t==3);const auto&s=a.getHistorySample(0);assert(s.hiddenTargetCsPrev[0]==action[0]&&s.behaviorProbPrev==p&&s.inputCs[0][0]==x[0]&&s.discount==discounts[t]);}
 assert(a.getLastReplayG()==0.f);int i=a.getLastReplayIndex();float q=0,g=1;for(int t=i;t>=0;t--){q+=a.getHistorySample(t).reward*g;g*=a.getHistorySample(t).discount;}assert(std::abs(q-a.getLastReplayQ())<1e-6);a.clearHistory();assert(a.getHistorySize()==0);std::cout<<"zeroTD, actual action/prob/state alignment, exact discount, terminal bootstrap, reset PASS\n";}
