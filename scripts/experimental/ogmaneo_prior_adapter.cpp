// Research adapter for OgmaNeo2 c5be9ec41d26aeeb2fc805961140781c851542cf.
// OgmaNeo sources retain their separate CC BY-NC-SA 4.0 license in vendor/.
#include "vendor/ogmaneo2_prior/ogmaneo/Hierarchy.h"
#include "vendor/ogmaneo2_prior/ogmaneo/ImageEncoder.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstring>
using namespace ogmaneo;
struct Adapter {
 ComputeSystem cs;ImageEncoder encoder;Hierarchy hierarchy;Actor actor;bool first=true;
 Adapter(unsigned seed){cs.rng.seed(seed);ComputeSystem::setNumThreads(1);
  ImageEncoder::VisibleLayerDesc iv;iv.size=Int3(16,16,7);iv.radius=2;encoder.initRandom(cs,Int3(8,8,16),{iv});
  std::vector<Hierarchy::LayerDesc> ld(3);for(int i=0;i<3;i++){ld[i].hiddenSize=Int3(8>>i,8>>i,16);ld[i].temporalHorizon=4;ld[i].ticksPerUpdate=2;}
  hierarchy.initRandom(cs,{Int3(8,8,16),Int3(1,1,3)},{InputType::prediction,InputType::none},ld);
  Actor::VisibleLayerDesc av;av.size=Int3(8,8,16);av.radius=8;actor.initRandom(cs,Int3(1,1,3),128,{av,av});
 }
 void reset(){hierarchy.clearState();actor.clearHistory();first=true;}
 std::vector<const IntBuffer*> context(){int t=hierarchy.getTicksPerUpdate(1)-1-hierarchy.getTicks(1);
  return {&hierarchy.getSCLayer(0).getHiddenCs(),&hierarchy.getPLayers(1)[t]->getHiddenCs()};}
};
extern "C" {
void* ogma_create(unsigned seed){try{return new Adapter(seed);}catch(...){return nullptr;}}
void ogma_free(void* ptr){delete static_cast<Adapter*>(ptr);}
void ogma_reset(void* ptr){static_cast<Adapter*>(ptr)->reset();}
int ogma_step(void* ptr,const float* input,int previousAction,const float* previousP,float reward,float discount,int terminal,int learnRepresentation,int learnActor,const float* prior,float* outP,float* outResidual){
 try {auto &a=*static_cast<Adapter*>(ptr);FloatBuffer x(input,input+1792);IntBuffer previous(1,previousAction);
  // First observation initializes caches; no fabricated cross-episode target.
  bool learn=learnRepresentation&&!a.first;
  a.encoder.step(a.cs,{&x},learn);a.hierarchy.step(a.cs,{&a.encoder.getHiddenCs(),&previous},learn,0.f,false);
  a.actor.setPrior(FloatBuffer(prior,prior+3));a.actor.setBehaviorProbPrev(FloatBuffer(previousP,previousP+3));
  a.actor.step(a.cs,a.context(),&previous,reward,learnActor&&!a.first,false,discount,terminal);
  a.first=false;for(int j=0;j<3;j++){outP[j]=a.actor.getOutputProbabilities()[j];outResidual[j]=a.actor.getOutputResidual()[j];if(!std::isfinite(outP[j])||!std::isfinite(outResidual[j]))return -2;}
  return a.actor.getHiddenCs()[0];
 }catch(...){return -1;}
}
int ogma_save(void* ptr,const char* path){try {auto &a=*static_cast<Adapter*>(ptr);std::ofstream f(path,std::ios::binary);a.encoder.writeToStream(f);a.hierarchy.writeToStream(f);a.actor.writeToStream(f);f<<a.cs.rng;return f.good()?0:-1;}catch(...){return -1;}}
int ogma_load(void* ptr,const char* path){try {auto &a=*static_cast<Adapter*>(ptr);std::ifstream f(path,std::ios::binary);a.encoder.readFromStream(f);a.hierarchy.readFromStream(f);a.actor.readFromStream(f);f>>a.cs.rng;a.reset();return !f.fail()?0:-1;}catch(...){return -1;}}
unsigned long long ogma_representation(void* ptr){auto &a=*static_cast<Adapter*>(ptr);unsigned long long hash=1469598103934665603ULL;
 auto append=[&](const std::vector<float>& v){auto p=reinterpret_cast<const unsigned char*>(v.data());for(size_t i=0;i<v.size()*sizeof(float);i++){hash^=p[i];hash*=1099511628211ULL;}};
 for(int j=0;j<a.encoder.getNumVisibleLayers();j++)append(a.encoder.getVisibleLayer(j).weights.nonZeroValues);
 for(int l=0;l<a.hierarchy.getNumLayers();l++){const auto &sc=a.hierarchy.getSCLayer(l);for(int j=0;j<sc.getNumVisibleLayers();j++)append(sc.getVisibleLayer(j).weights.nonZeroValues);for(const auto &p:a.hierarchy.getPLayers(l))if(p)for(int j=0;j<p->getNumVisibleLayers();j++)append(p->getVisibleLayer(j).weights.nonZeroValues);}
 return hash;
}
unsigned long long ogma_weights(void* ptr){auto &a=*static_cast<Adapter*>(ptr);unsigned long long hash=ogma_representation(ptr);auto append=[&](const std::vector<float>&v){auto p=reinterpret_cast<const unsigned char*>(v.data());for(size_t i=0;i<v.size()*sizeof(float);i++){hash^=p[i];hash*=1099511628211ULL;}};
 for(int j=0;j<a.actor.getNumVisibleLayers();j++){append(a.actor.getVisibleLayer(j).valueWeights.nonZeroValues);append(a.actor.getVisibleLayer(j).actionWeights.nonZeroValues);}return hash;
}
unsigned long long ogma_actor(void* ptr){auto &a=*static_cast<Adapter*>(ptr);unsigned long long hash=1469598103934665603ULL;auto append=[&](const std::vector<float>&v){auto p=reinterpret_cast<const unsigned char*>(v.data());for(size_t i=0;i<v.size()*sizeof(float);i++){hash^=p[i];hash*=1099511628211ULL;}};for(int j=0;j<a.actor.getNumVisibleLayers();j++){append(a.actor.getVisibleLayer(j).valueWeights.nonZeroValues);append(a.actor.getVisibleLayer(j).actionWeights.nonZeroValues);}return hash;}
int ogma_zero_actor(void* ptr){auto &a=*static_cast<Adapter*>(ptr);for(int j=0;j<a.actor.getNumVisibleLayers();j++){for(float v:a.actor.getVisibleLayer(j).actionWeights.nonZeroValues)if(v!=0.f)return 0;}return 1;}
}
